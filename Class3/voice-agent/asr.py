import tempfile
from faster_whisper import WhisperModel

asr_model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe(audio_bytes: bytes) -> str:
    # Save uploaded bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Transcribe
    segments, _ = asr_model.transcribe(tmp_path)
    text = "".join(segment.text for segment in segments)

    return text.strip()