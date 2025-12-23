
import os
import sys
import shutil
import logging
import tempfile
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

print("=== PROGRAMMSTART ===", flush=True)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
level=logging.INFO,
format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AutoPipeline")

# -------------------------
# Optional Imports
# -------------------------
try:
import whisper
except Exception:
whisper = None

try:
import cv2
except Exception:
cv2 = None

try:
from deepface import DeepFace
except Exception:
DeepFace = None

try:
from moviepy.editor import VideoFileClip, concatenate_videoclips
except Exception as e:
print("Fehler beim Importieren von moviepy:", e)
sys.exit(1)

# -------------------------
# Parameter
# -------------------------
HIGHLIGHT_PRE_PADDING = 1.0
HIGHLIGHT_POST_PADDING = 2.0
MIN_HIGHLIGHT_LENGTH = 1.0
EMOTION_CONF_THRESHOLD = 0.5
FACE_FRAME_SKIP = 5

# -------------------------
# FFmpeg Check
# -------------------------
def ensure_ffmpeg_available():
import subprocess

try:
subprocess.run(
["ffmpeg", "-version"],
stdout=subprocess.DEVNULL,
stderr=subprocess.DEVNULL,
check=True
)
logger.info("FFmpeg über Systempfad verfügbar")
return
except Exception:
pass

ffmpeg_folder = Path(r"C:\Users\Computer\AppData\Local\Microsoft\WinGet\Links")
ffmpeg_exe = ffmpeg_folder / "ffmpeg.exe"
ffprobe_exe = ffmpeg_folder / "ffprobe.exe"

if ffmpeg_exe.exists() and ffprobe_exe.exists():
os.environ["FFMPEG_BINARY"] = str(ffmpeg_exe)
os.environ["FFPROBE_BINARY"] = str(ffprobe_exe)
logger.info(f"FFmpeg lokal gefunden: {ffmpeg_exe}")
return

raise RuntimeError("FFmpeg nicht gefunden (Systempfad oder WinGet)")

# -------------------------
# Whisper
# -------------------------
def transcribe_with_local_whisper(
video_path: str,
model_size: str = "small"
) -> Dict[str, Any]:
if whisper is None:
raise RuntimeError("whisper ist nicht installiert")

logger.info("Lade Whisper-Modell (%s)", model_size)
model = whisper.load_model(model_size)
result = model.transcribe(video_path)
logger.info("Whisper-Transkription abgeschlossen")
return result

# -------------------------
# Emotion Detection
# -------------------------
def detect_emotion_segments(video_path: str) -> List[Tuple[float, float]]:
if cv2 is None or DeepFace is None:
logger.warning("Emotionserkennung deaktiviert")
return []

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

segments = []
frame_index = 0
last_hit_time = None

while True:
ret, frame = cap.read()
if not ret:
break

if frame_index % FACE_FRAME_SKIP == 0:
t = frame_index / fps
try:
analysis = DeepFace.analyze(
frame,
actions=["emotion"],
enforce_detection=False
)
if isinstance(analysis, list):
analysis = analysis[0]

emotions = analysis.get("emotion", {})
if emotions and max(emotions.values()) >= EMOTION_CONF_THRESHOLD:
if last_hit_time is None or t - last_hit_time > 1.0:
segments.append((
max(0, t - HIGHLIGHT_PRE_PADDING),
min(duration, t + HIGHLIGHT_POST_PADDING)
))
last_hit_time = t
except Exception:
pass

frame_index += 1

cap.release()
return merge_close_segments(segments)

# -------------------------
# Segment Merge
# -------------------------
def merge_close_segments(
segments: List[Tuple[float, float]],
gap: float = 1.0
) -> List[Tuple[float, float]]:
if not segments:
return []

segments.sort()
merged = [segments[0]]

for s, e in segments[1:]:
last_s, last_e = merged[-1]
if s <= last_e + gap:
merged[-1] = (last_s, max(last_e, e))
else:
merged.append((s, e))

return merged

# -------------------------
# Video Creation
# -------------------------
def create_highlight_clip(
source_video: str,
highlights: List[Tuple[float, float]],
out_path: str
):
with VideoFileClip(source_video) as vid:
clips = [
vid.subclip(s, e)
for s, e in highlights
if e - s >= MIN_HIGHLIGHT_LENGTH
]

if not clips:
logger.warning("Keine gültigen Highlight-Clips")
return

final = concatenate_videoclips(clips, method="compose")

with tempfile.NamedTemporaryFile(
suffix=".mp4",
delete=False
) as tmp:
tmp_path = tmp.name

final.write_videofile(
tmp_path,
codec="libx264",
audio_codec="aac",
fps=vid.fps
)

final.close()
shutil.move(tmp_path, out_path)

# -------------------------
# File Processing
# -------------------------
def process_one_file(file_path: Path, out_folder: Path):
logger.info("Verarbeite: %s", file_path)

try:
emotion_segments = detect_emotion_segments(str(file_path))

if emotion_segments:
highlights = emotion_segments
else:
logger.warning("Keine Emotionen erkannt – Fallback")
with VideoFileClip(str(file_path)) as v:
highlights = [(0, min(10, v.duration))]

out_folder.mkdir(parents=True, exist_ok=True)
out_video = out_folder / f"{file_path.stem}_highlight.mp4"

create_highlight_clip(
str(file_path),
highlights,
str(out_video)
)

logger.info("Fertig: %s", out_video)

except Exception:
logger.exception("Fehler bei %s", file_path)

# -------------------------
# Main
# -------------------------
def main():
parser = argparse.ArgumentParser()
parser.add_argument("--watch-folder", required=True)
parser.add_argument("--out-folder", required=True)
args = parser.parse_args()

ensure_ffmpeg_available()

watch = Path(args.watch_folder)
out = Path(args.out_folder)

for file in watch.iterdir():
if file.suffix.lower() in (".mp4", ".mkv", ".mov"):
process_one_file(file, out)

if __name__ == "__main__":
main()






