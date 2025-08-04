import os
import subprocess
import whisper
import json
import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Paths
AUDIO_DIR = "/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_Bonus3/source-audio"
TRANSCRIPT_DIR = "/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_Bonus3/transripts"

# YouTube video URLs
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=dPXKPSlMygw",
    "https://www.youtube.com/watch?v=snkiY6_0gN8",
    "https://www.youtube.com/watch?v=yWytJp3x28U"
]

# Ensure output directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

def already_downloaded(title):
    """Check if audio already exists in source-audio directory."""
    candidates = glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
    for file in candidates:
        if title.lower() in Path(file).stem.lower():
            return file
    return None

def download_audio(url):
    """Download audio using yt-dlp and return the path to the file."""
    print(f"\n▶️  Checking audio for {url}")
    result = subprocess.run([
        "yt-dlp",
        "--get-title",
        url
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Failed to get title: {result.stderr}")
        return None

    title = result.stdout.strip()
    existing = already_downloaded(title)
    if existing:
        print(f"⏩ Already downloaded: {existing}")
        return existing

    print(f"⬇️  Downloading: {title}")
    subprocess.run([
        "yt-dlp",
        "-f", "bestaudio",
        "-x", "--audio-format", "mp3",
        "-o", f"{AUDIO_DIR}/%(title)s.%(ext)s",
        url
    ], check=True)

    # Get downloaded file path
    matching_files = glob.glob(f"{AUDIO_DIR}/{title}*.mp3")
    if matching_files:
        print(f"✅ Downloaded to: {matching_files[0]}")
        return matching_files[0]
    else:
        print(f"⚠️  Couldn't find downloaded file for {title}")
        return None

def transcribe_audio_to_jsonl(audio_path):
    """Transcribe audio using Whisper and save as JSONL with timestamps."""
    print(f"🧠 Transcribing: {audio_path}")
    model = whisper.load_model("base")

    try:
        result = model.transcribe(audio_path, verbose=False, language="en")
    except Exception as e:
        print(f"❌ Failed to transcribe {audio_path}: {e}")
        return

    filename = os.path.splitext(os.path.basename(audio_path))[0]
    jsonl_path = os.path.join(TRANSCRIPT_DIR, f"{filename}.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            entry = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"📄 Saved transcript to: {jsonl_path}")

def main():
    # Phase 1: Download all audio
    audio_paths = []
    for url in YOUTUBE_URLS:
        path = download_audio(url)
        if path:
            audio_paths.append(path)

    # Phase 2: Transcribe in parallel
    print(f"\n⚙️  Transcribing {len(audio_paths)} files using {cpu_count()} cores...")
    with Pool(processes=cpu_count()) as pool:
        pool.map(transcribe_audio_to_jsonl, audio_paths)

if __name__ == "__main__":
    main()

