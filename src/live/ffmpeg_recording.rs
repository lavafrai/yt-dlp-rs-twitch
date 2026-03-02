//! FFmpeg-based live stream recorder (fallback).
//!
//! Uses FFmpeg's native HLS support to record a live stream to a file.
//! This is the fallback engine when reqwest-based recording is not suitable
//! (e.g., encrypted streams, complex HLS features, or user preference).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::sync::CancellationToken;

use crate::error::Result;
use crate::events::DownloadEvent;
use crate::events::types::RecordingMethod;
use crate::executor::Executor;
use crate::live::{LiveProgress, ProgressCallback};

/// Parsed statistics extracted from one FFmpeg stderr progress line.
struct FfmpegStats {
    /// Total bytes written to the output file so far (accumulated).
    bytes: u64,
    /// Current encoding / copy bitrate in bits per second.
    bitrate_bps: f64,
}

/// Reads FFmpeg stderr line-by-line and emits [`DownloadEvent::LiveRecordingProgress`] events.
///
/// Runs inside a detached `tokio::spawn` task until the stderr pipe closes (process exits).
async fn read_ffmpeg_progress(
    stderr: tokio::process::ChildStderr,
    video_id: String,
    start: Instant,
    progress_callback: Option<ProgressCallback>,
    event_bus: crate::events::EventBus,
) {
    let mut reader = BufReader::new(stderr);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                if let Some(stats) = parse_ffmpeg_stats_line(line.trim()) {
                    let progress = LiveProgress {
                        bytes_written: stats.bytes,
                        elapsed: start.elapsed(),
                        bitrate_bps: stats.bitrate_bps,
                        segments: 0,
                    };
                    if let Some(cb) = &progress_callback {
                        cb.call(progress.clone());
                    }
                    event_bus.emit_if_subscribed(DownloadEvent::LiveRecordingProgress {
                        video_id: video_id.clone(),
                        elapsed: progress.elapsed,
                        bytes_written: progress.bytes_written,
                        segments: progress.segments,
                        bitrate_bps: progress.bitrate_bps,
                    });
                }
            }
        }
    }
}

/// Parses a single FFmpeg stats stderr line.
///
/// Matches lines that contain both `size=` and `time=`, e.g.:
/// `size=  12288kB time=00:00:41.16 bitrate=2444.1kbits/s speed=1.00x`
///
/// Returns `None` for all other lines (info, errors, etc.).
fn parse_ffmpeg_stats_line(line: &str) -> Option<FfmpegStats> {
    if !line.contains("size=") || !line.contains("time=") {
        return None;
    }
    let bytes = parse_ffmpeg_size(line)?;
    let bitrate_bps = parse_ffmpeg_bitrate(line).unwrap_or(0.0);
    Some(FfmpegStats { bytes, bitrate_bps })
}

/// Extracts the `size=` field value and converts it to bytes.
fn parse_ffmpeg_size(line: &str) -> Option<u64> {
    let start = line.find("size=")? + 5;
    let rest = line[start..].trim_start();
    let num_end = rest.find(|c: char| !c.is_ascii_digit())?;
    let n: u64 = rest[..num_end].parse().ok()?;
    let suffix = rest[num_end..].trim_start().to_ascii_lowercase();
    if suffix.starts_with('m') {
        Some(n * 1024 * 1024)
    } else {
        // Default unit is kB
        Some(n * 1024)
    }
}

/// Extracts the `bitrate=` field value and converts it to bits per second.
fn parse_ffmpeg_bitrate(line: &str) -> Option<f64> {
    let start = line.find("bitrate=")? + 8;
    let rest = line[start..].trim_start();
    let num_end = rest.find(|c: char| !c.is_ascii_digit() && c != '.')?;
    let n: f64 = rest[..num_end].parse().ok()?;
    let suffix = rest[num_end..].to_ascii_lowercase();
    // "kbits/s" → ×1 000; "Mbits/s" → ×1 000 000
    let multiplier = if suffix.starts_with('m') { 1_000_000.0 } else { 1_000.0 };
    Some(n * multiplier)
}

/// FFmpeg-based live stream recorder.
///
/// Spawns an FFmpeg process with `-i <hls_url> -c copy <output>` and
/// manages its lifecycle via stdin (`q` for graceful stop) or kill.
#[derive(Debug)]
pub struct FfmpegLiveRecorder {
    /// The HLS stream URL to record.
    stream_url: String,
    /// The output file path.
    output_path: PathBuf,
    /// Path to the FFmpeg binary.
    ffmpeg_path: PathBuf,
    /// The video ID (for event emission).
    video_id: String,
    /// Optional maximum recording duration.
    max_duration: Option<Duration>,
    /// Whether to start from the beginning of the stream buffer.
    live_from_start: bool,
    /// Optional direct progress callback.
    progress_callback: Option<ProgressCallback>,
    /// Cancellation token for graceful stop.
    cancellation_token: CancellationToken,
    /// The event bus for emitting recording events.
    event_bus: crate::events::EventBus,
    /// Quality label for event metadata.
    quality: String,
}

impl FfmpegLiveRecorder {
    /// Creates a new `FfmpegLiveRecorder`.
    ///
    /// # Arguments
    ///
    /// * `stream_url` - The HLS stream URL to record.
    /// * `output_path` - Where to write the recorded stream.
    /// * `ffmpeg_path` - Path to the FFmpeg binary.
    /// * `video_id` - The video ID (for events).
    /// * `quality` - Quality label (e.g. "1080p").
    /// * `max_duration` - Optional maximum recording duration.
    /// * `live_from_start` - When `true`, passes `-live_start_index 0` to FFmpeg
    ///   so recording starts from the first available HLS segment instead of
    ///   the default near-live position.
    /// * `progress_callback` - Optional callback invoked on every progress line from stderr.
    /// * `cancellation_token` - Token to cancel recording.
    /// * `event_bus` - Event bus for broadcasting progress.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stream_url: impl Into<String>,
        output_path: impl Into<PathBuf>,
        ffmpeg_path: impl Into<PathBuf>,
        video_id: impl Into<String>,
        quality: impl Into<String>,
        max_duration: Option<Duration>,
        live_from_start: bool,
        progress_callback: Option<ProgressCallback>,
        cancellation_token: CancellationToken,
        event_bus: crate::events::EventBus,
    ) -> Self {
        Self {
            stream_url: stream_url.into(),
            output_path: output_path.into(),
            ffmpeg_path: ffmpeg_path.into(),
            video_id: video_id.into(),
            quality: quality.into(),
            max_duration,
            live_from_start,
            progress_callback,
            cancellation_token,
            event_bus,
        }
    }

    /// Starts the FFmpeg recording.
    ///
    /// Spawns FFmpeg as a long-running process and waits for it to finish.
    /// The process is stopped when the cancellation token is triggered or
    /// the max duration is reached.
    ///
    /// # Errors
    ///
    /// Returns an error if FFmpeg cannot be spawned, exits with an error,
    /// or the output file cannot be accessed.
    ///
    /// # Returns
    ///
    /// A [`super::RecordingResult`] with recording statistics.
    pub async fn record(&self) -> Result<super::RecordingResult> {
        let start = Instant::now();

        tracing::info!(
            url = self.stream_url,
            video_id = self.video_id,
            output = ?self.output_path,
            max_duration = ?self.max_duration,
            "📥 Starting live recording (ffmpeg)"
        );

        self.event_bus.emit_if_subscribed(DownloadEvent::LiveRecordingStarted {
            video_id: self.video_id.clone(),
            url: self.stream_url.clone(),
            quality: self.quality.clone(),
            method: RecordingMethod::Fallback,
        });

        // Ensure output directory exists
        if let Some(parent) = self.output_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Build FFmpeg args: [-live_start_index 0] -i <url> -c copy [-t duration] -y <output>
        let mut args: Vec<String> = Vec::new();

        if self.live_from_start {
            // Start from the first available HLS segment instead of near-live default
            args.push("-live_start_index".to_string());
            args.push("0".to_string());
        }

        args.extend([
            "-i".to_string(),
            self.stream_url.clone(),
            "-c".to_string(),
            "copy".to_string(),
        ]);

        if let Some(max) = self.max_duration {
            args.push("-t".to_string());
            args.push(max.as_secs().to_string());
        }

        args.push("-y".to_string());
        args.push(self.output_path.display().to_string());

        let executor = Executor::new(&self.ffmpeg_path, &args, Duration::from_secs(0));
        let mut process = executor.execute_streaming().await?;

        // Spawn a task that reads FFmpeg stderr stats and emits LiveRecordingProgress events.
        // Must happen before the select! loop so the reader runs concurrently.
        if let Some(stderr) = process.take_stderr() {
            let event_bus = self.event_bus.clone();
            let video_id = self.video_id.clone();
            let cb = self.progress_callback.clone();
            tokio::spawn(read_ffmpeg_progress(stderr, video_id, start, cb, event_bus));
        }

        // Wait for cancellation or process exit
        let mut exit_error: Option<crate::error::Error> = None;
        let stop_reason = tokio::select! {
            _ = self.cancellation_token.cancelled() => {
                tracing::info!(video_id = self.video_id, "📥 Cancellation requested, stopping ffmpeg");
                match process.stop().await {
                    Ok(_) => "cancelled".to_string(),
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to stop ffmpeg gracefully, killing");
                        let _ = process.kill().await;
                        "cancelled (killed)".to_string()
                    }
                }
            }
            result = process.wait() => {
                match result {
                    Ok(output) if output.code == 0 => "stream ended".to_string(),
                    Ok(output) => {
                        let reason = format!(
                            "ffmpeg exited with code {}: {}",
                            output.code,
                            output.stderr.lines().last().unwrap_or("")
                        );
                        tracing::warn!(video_id = self.video_id, exit_code = output.code, "FFmpeg exited with non-zero code");
                        exit_error = Some(crate::error::Error::CommandFailed {
                            command: "ffmpeg".to_string(),
                            exit_code: output.code,
                            stderr: output.stderr,
                        });
                        reason
                    }
                    Err(e) => {
                        let reason = format!("ffmpeg process error: {e}");
                        tracing::warn!(video_id = self.video_id, error = %e, "FFmpeg process error");
                        exit_error = Some(e);
                        reason
                    }
                }
            }
        };

        let total_duration = start.elapsed();

        // Get actual file size
        let total_bytes = tokio::fs::metadata(&self.output_path)
            .await
            .map(|m| m.len())
            .unwrap_or(0);

        tracing::info!(
            video_id = self.video_id,
            total_bytes = total_bytes,
            duration = ?total_duration,
            reason = stop_reason,
            "✅ FFmpeg live recording stopped"
        );

        self.event_bus.emit_if_subscribed(DownloadEvent::LiveRecordingStopped {
            video_id: self.video_id.clone(),
            reason: stop_reason,
            output_path: self.output_path.clone(),
            total_bytes,
            total_duration,
        });

        if let Some(err) = exit_error {
            return Err(err);
        }

        Ok(super::RecordingResult {
            output_path: self.output_path.clone(),
            total_bytes,
            total_duration,
            segments_downloaded: 0, // ffmpeg handles segments internally
        })
    }
}

impl std::fmt::Display for FfmpegLiveRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FfmpegLiveRecorder(video_id={}, quality={}, output={})",
            self.video_id,
            self.quality,
            self.output_path.display()
        )
    }
}
