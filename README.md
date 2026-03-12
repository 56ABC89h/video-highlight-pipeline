import time
import shutil
import tempfile
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yt_dlp

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from moviepy import VideoFileClip, concatenate_videoclips


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ShortsBot")

print("=== PROGRAMMSTART ===", flush=True)


# =========================
# PARAMETER
# =========================
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv")

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

MIN_HIGHLIGHT_LENGTH = 5.0
MAX_HIGHLIGHT_LENGTH = 30.0

WINDOW_SIZE = 0.5
THRESHOLD_FACTOR = 1.3

# YouTube Kanal
CHANNEL_URL = "https://www.youtube.com/@CHANNELNAME/videos"

# wie oft der Kanal geprüft wird (Sekunden)
CHECK_INTERVAL = 300


# =========================
# YOUTUBE DOWNLOAD
# =========================
def download_channel_videos(channel_url: str, input_dir: Path):

    ydl_opts = {
        "outtmpl": str(input_dir / "%(title)s.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "download_archive": "downloaded.txt",
        "quiet": True
    }

    logger.info("Prüfe YouTube Kanal auf neue Videos...")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([channel_url])

    logger.info("Download abgeschlossen")


# =========================
# VIDEO NORMALISIERUNG
# =========================
def normalize_video(input_path: str) -> str:

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        normalized_path = tmp.name

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", "fps=30",
        "-vsync", "cfr",
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        "-ar", "48000",
        normalized_path
    ]

    logger.info("Normalisiere Video (CFR + 48kHz Audio)")

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return normalized_path


# =========================
# AUDIO HIGHLIGHT ANALYSE
# =========================
def detect_audio_highlights(video_path: str) -> List[Tuple[float, float]]:
    highlights = []

    with VideoFileClip(video_path) as clip:

        if clip.audio is None:
            logger.warning("Kein Audio – keine Analyse möglich")
            return []

        audio = clip.audio.to_soundarray()
        audio = audio.mean(axis=1)

        audio_fps = clip.audio.fps
        step = int(WINDOW_SIZE * audio_fps)

        volumes = []
        times = []

        for i in range(0, len(audio), step):

            window = audio[i:i + step]

            if len(window) == 0:
                continue

            rms = np.sqrt(np.mean(window ** 2))

            volumes.append(rms)
            times.append(i / audio_fps)

        if not volumes:
            return []

        threshold = np.mean(volumes) * THRESHOLD_FACTOR

        active = False
        start = 0.0

        for t, v in zip(times, volumes):

            if v >= threshold and not active:
                active = True
                start = t

            elif v < threshold and active:

                end = t
                duration = end - start

                if MIN_HIGHLIGHT_LENGTH <= duration <= MAX_HIGHLIGHT_LENGTH:
                    highlights.append((start, end))

                active = False

        if active:

            end = clip.duration
            duration = end - start

            if MIN_HIGHLIGHT_LENGTH <= duration <= MAX_HIGHLIGHT_LENGTH:
                highlights.append((start, end))

    logger.info("Audio-Highlights erkannt: %d", len(highlights))

    return highlights


# =========================
# SHORTS FORMAT
# =========================
def to_shorts_format(clip: VideoFileClip) -> VideoFileClip:

    w, h = clip.size
    target_ratio = 9 / 16
    current_ratio = w / h

    if current_ratio > target_ratio:

        new_width = int(h * target_ratio)
        x1 = (w - new_width) // 2

        clip = clip.cropped(x1=x1, width=new_width)

    elif current_ratio < target_ratio:

        new_height = int(w / target_ratio)
        y1 = (h - new_height) // 2

        clip = clip.cropped(y1=y1, height=new_height)

    return clip.resized((1080, 1920))


# =========================
# HIGHLIGHT CLIP ERSTELLEN
# =========================
def create_highlight_clip(
    source_video: str,
    highlights: List[Tuple[float, float]],
    out_path: str
):

    logger.info("Erstelle Highlight-Clip")

    processed_clips = []

    with VideoFileClip(source_video) as video:

        for start, end in highlights:

            if end - start < MIN_HIGHLIGHT_LENGTH:
                continue

            sub = video.subclip(start, end)
            sub = to_shorts_format(sub)

            processed_clips.append(sub)

        if not processed_clips:
            logger.warning("Keine gültigen Clips → Abbruch")
            return

        final = concatenate_videoclips(processed_clips, method="compose")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        final.write_videofile(
            tmp_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            audio_fps=48000,
            threads=4,
            ffmpeg_params=["-async", "1"]
        )

        final.close()

    shutil.move(tmp_path, out_path)

    logger.info("Gespeichert: %s", out_path)


# =========================
# VIDEO VERARBEITEN
# =========================
def process_video(video_path: Path, output_dir: Path):

    logger.info("Verarbeite: %s", video_path.name)

    normalized_video = normalize_video(str(video_path))

    highlights = detect_audio_highlights(normalized_video)

    if not highlights:

        logger.warning("Fallback aktiv: erste 10 Sekunden")

        with VideoFileClip(normalized_video) as clip:
            end = min(10, clip.duration)

        highlights = [(0, end)]

    out_file = output_dir / f"{video_path.stem}_shorts.mp4"

    create_highlight_clip(
        normalized_video,
        highlights,
        str(out_file)
    )

    try:
        Path(normalized_video).unlink()
    except:
        pass


# =========================
# WATCHDOG
# =========================
class VideoHandler(FileSystemEventHandler):

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def on_created(self, event):

        if event.is_directory:
            return

        path = Path(event.src_path)

        if path.suffix.lower() in VIDEO_EXTENSIONS:

            logger.info("Neue Datei erkannt: %s", path.name)

            time.sleep(2)

            process_video(path, self.output_dir)


# =========================
# MAIN
# =========================
def main():

    input_dir = Path("input")
    output_dir = Path("output")

    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    logger.info("Input:  %s", input_dir.resolve())
    logger.info("Output: %s", output_dir.resolve())

    observer = Observer()

    observer.schedule(
        VideoHandler(output_dir),
        str(input_dir),
        recursive=False
    )

    observer.start()

    logger.info("Bot läuft und überwacht Kanal...")

    try:

        while True:

            download_channel_videos(CHANNEL_URL, input_dir)

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:

        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()