//! yt-dlp-based live stream recorder with `--live-from-start` support.
//!
//! Delegates to the yt-dlp binary for recording, which handles platform-specific
//! logic for fetching the stream from the very beginning (e.g. Twitch VOD replay,
//! YouTube DVR). This is the only reliable way to implement `--live-from-start`
//! across all yt-dlp-supported sites.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;

use crate::error::Result;
use crate::events::DownloadEvent;
use crate::events::types::RecordingMethod;
use crate::executor::Executor;
use crate::live::{LiveProgress, ProgressCallback};

/// Interval between file-size polls for progress reporting.
const PROGRESS_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Periodically polls the output file size and emits progress events.
///
/// Runs inside a detached `tokio::spawn` task. Checks the file size every
/// [`PROGRESS_POLL_INTERVAL`] and reports via the optional callback and the
/// event bus. Stops when `cancel` is triggered.
///
/// This approach is more reliable than parsing yt-dlp output because yt-dlp
/// may delegate to ffmpeg or other external downloaders internally, bypassing
/// its own progress hooks.
async fn poll_file_progress(
    output_path: PathBuf,
    video_id: String,
    start: Instant,
    progress_callback: Option<ProgressCallback>,
    event_bus: crate::events::EventBus,
    cancel: CancellationToken,
) {
    let mut interval = tokio::time::interval(PROGRESS_POLL_INTERVAL);
    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            _ = interval.tick() => {
                let bytes = tokio::fs::metadata(&output_path)
                    .await
                    .map(|m| m.len())
                    .unwrap_or(0);
                if bytes == 0 {
                    continue;
                }
                let elapsed = start.elapsed();
                let bitrate_bps = if elapsed.as_secs_f64() > 0.0 {
                    (bytes as f64 * 8.0) / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                let progress = LiveProgress {
                    bytes_written: bytes,
                    elapsed,
                    bitrate_bps,
                    segments: 0,
                };
                if let Some(cb) = &progress_callback {
                    cb.call(progress.clone());
                }
                event_bus.emit_if_subscribed(DownloadEvent::LiveRecordingProgress {
                    video_id: video_id.clone(),
                    elapsed,
                    bytes_written: bytes,
                    segments: 0,
                    bitrate_bps,
                });
            }
        }
    }
}

/// yt-dlp-based live stream recorder.
///
/// Spawns yt-dlp with `--live-from-start` and manages its lifecycle.
/// This recorder delegates all platform-specific DVR/replay logic to yt-dlp,
/// making it the only engine capable of true "from start" recording.
#[derive(Debug)]
pub struct YtDlpLiveRecorder {
    /// The live stream URL to record.
    stream_url: String,
    /// The output file path.
    output_path: PathBuf,
    /// Path to the yt-dlp binary.
    ytdlp_path: PathBuf,
    /// The video ID (for event emission).
    video_id: String,
    /// Extra user arguments to pass to yt-dlp.
    extra_args: Vec<String>,
    /// Optional maximum recording duration.
    max_duration: Option<Duration>,
    /// Optional direct progress callback.
    progress_callback: Option<ProgressCallback>,
    /// Cancellation token for graceful stop.
    cancellation_token: CancellationToken,
    /// The event bus for emitting recording events.
    event_bus: crate::events::EventBus,
    /// Quality label for event metadata.
    quality: String,
}

impl YtDlpLiveRecorder {
    /// Creates a new `YtDlpLiveRecorder`.
    ///
    /// # Arguments
    ///
    /// * `stream_url` - The live stream URL to record (original webpage URL).
    /// * `output_path` - Where to write the recorded stream.
    /// * `ytdlp_path` - Path to the yt-dlp binary.
    /// * `video_id` - The video ID (for events).
    /// * `quality` - Quality label (e.g. "1080p").
    /// * `extra_args` - Additional yt-dlp arguments (cookies, proxy, etc.).
    /// * `max_duration` - Optional maximum recording duration.
    /// * `progress_callback` - Optional callback invoked on every progress line from stderr.
    /// * `cancellation_token` - Token to cancel recording.
    /// * `event_bus` - Event bus for broadcasting progress.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stream_url: impl Into<String>,
        output_path: impl Into<PathBuf>,
        ytdlp_path: impl Into<PathBuf>,
        video_id: impl Into<String>,
        quality: impl Into<String>,
        extra_args: Vec<String>,
        max_duration: Option<Duration>,
        progress_callback: Option<ProgressCallback>,
        cancellation_token: CancellationToken,
        event_bus: crate::events::EventBus,
    ) -> Self {
        Self {
            stream_url: stream_url.into(),
            output_path: output_path.into(),
            ytdlp_path: ytdlp_path.into(),
            video_id: video_id.into(),
            quality: quality.into(),
            extra_args,
            max_duration,
            progress_callback,
            cancellation_token,
            event_bus,
        }
    }

    /// Starts the yt-dlp recording with `--live-from-start`.
    ///
    /// Spawns yt-dlp as a long-running process and waits for it to finish.
    /// The process is killed when the cancellation token is triggered or
    /// the max duration is reached.
    ///
    /// # Errors
    ///
    /// Returns an error if yt-dlp cannot be spawned, exits with an error,
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
            "📥 Starting live recording (yt-dlp --live-from-start)"
        );

        self.event_bus
            .emit_if_subscribed(DownloadEvent::LiveRecordingStarted {
                video_id: self.video_id.clone(),
                url: self.stream_url.clone(),
                quality: self.quality.clone(),
                method: RecordingMethod::YtDlp,
            });

        // Ensure output directory exists
        if let Some(parent) = self.output_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Build yt-dlp args:
        //   --live-from-start --no-part -o <output> [extra_args...] <url>
        //
        // We intentionally omit --progress / --progress-template because yt-dlp
        // may delegate downloading to ffmpeg internally (e.g. Twitch), bypassing
        // its own progress hooks. Progress is instead tracked by polling the
        // output file size.
        let mut args: Vec<String> = vec![
            "--live-from-start".to_string(),
            "--no-part".to_string(),
            "-o".to_string(),
            self.output_path.display().to_string(),
        ];

        args.extend(self.extra_args.clone());
        args.push(self.stream_url.clone());

        // Use timeout = 0 since StreamingProcess handles its own lifecycle
        let executor = Executor::new(&self.ytdlp_path, &args, Duration::from_secs(0));
        let mut process = executor.execute_streaming().await?;

        // Spawn a task that polls the output file size for progress reporting.
        // This is more reliable than parsing yt-dlp output because yt-dlp may
        // delegate downloading to ffmpeg or other backends internally.
        let progress_cancel = CancellationToken::new();
        {
            let output_path = self.output_path.clone();
            let video_id = self.video_id.clone();
            let cb = self.progress_callback.clone();
            let event_bus = self.event_bus.clone();
            let cancel = progress_cancel.clone();
            tokio::spawn(poll_file_progress(
                output_path, video_id, start, cb, event_bus, cancel,
            ));
        }

        // Wait for cancellation, max duration, or process exit
        let mut exit_error: Option<crate::error::Error> = None;
        let stop_reason = tokio::select! {
            _ = self.cancellation_token.cancelled() => {
                tracing::info!(video_id = self.video_id, "📥 Cancellation requested, stopping yt-dlp");
                let _ = process.kill().await;
                "cancelled".to_string()
            }
            _ = Self::wait_max_duration(self.max_duration) => {
                tracing::info!(video_id = self.video_id, "📥 Max duration reached, stopping yt-dlp");
                let _ = process.kill().await;
                "max duration reached".to_string()
            }
            result = process.wait() => {
                match result {
                    Ok(output) if output.code == 0 => "stream ended".to_string(),
                    Ok(output) => {
                        let reason = format!(
                            "yt-dlp exited with code {}: {}",
                            output.code,
                            output.stderr.lines().last().unwrap_or("")
                        );
                        tracing::warn!(
                            video_id = self.video_id,
                            exit_code = output.code,
                            "yt-dlp exited with non-zero code"
                        );
                        exit_error = Some(crate::error::Error::CommandFailed {
                            command: "yt-dlp".to_string(),
                            exit_code: output.code,
                            stderr: output.stderr,
                        });
                        reason
                    }
                    Err(e) => {
                        let reason = format!("yt-dlp process error: {e}");
                        tracing::warn!(
                            video_id = self.video_id,
                            error = %e,
                            "yt-dlp process error"
                        );
                        exit_error = Some(e);
                        reason
                    }
                }
            }
        };

        // Stop the progress polling task
        progress_cancel.cancel();

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
            "✅ yt-dlp live recording stopped"
        );

        self.event_bus
            .emit_if_subscribed(DownloadEvent::LiveRecordingStopped {
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
            segments_downloaded: 0,
        })
    }

    /// Waits for the max duration, or forever if `None`.
    async fn wait_max_duration(max_duration: Option<Duration>) {
        match max_duration {
            Some(d) => tokio::time::sleep(d).await,
            None => std::future::pending().await,
        }
    }
}

impl std::fmt::Display for YtDlpLiveRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "YtDlpLiveRecorder(video_id={}, quality={}, output={})",
            self.video_id,
            self.quality,
            self.output_path.display()
        )
    }
}
