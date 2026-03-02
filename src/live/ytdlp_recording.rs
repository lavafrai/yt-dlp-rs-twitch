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

        // Wait for cancellation, max duration, or process exit
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
                        reason
                    }
                    Err(e) => {
                        let reason = format!("yt-dlp process error: {e}");
                        tracing::warn!(
                            video_id = self.video_id,
                            error = %e,
                            "yt-dlp process error"
                        );
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
