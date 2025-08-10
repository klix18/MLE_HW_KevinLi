# tts.py
import subprocess
import tempfile
from pathlib import Path

def synthesize_speech(text: str) -> str:
    text = (text or "").strip() or "Hello, this is a test."

    # 1) Generate AIFF using macOS 'say'
    aiff_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".aiff").name)
    subprocess.run(["say", "-o", str(aiff_path), text], check=True)

    # 2) Convert AIFF -> WAV using ffmpeg (installed for Whisper)
    wav_path = aiff_path.with_suffix(".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff_path), str(wav_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    return str(wav_path)

