from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import asr
import llm
import tts
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import io, os, re


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse(Path("static/index.html"))

@app.get("/history")
def history():
    return JSONResponse(llm.get_history())
    
@app.get("/llm-stream")
def llm_stream(q: str):
    def gen():
        for chunk in llm.stream_response(q):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/asr-chunk")
async def asr_chunk(file: UploadFile = File(...)):
  # very small chunks, so keep it lightweight; no history here
  audio_bytes = await file.read()
  if not audio_bytes:
      raise HTTPException(status_code=400, detail="empty chunk")
  try:
      # reuse your asr.transcribe to keep code simple:
      # (it transcribes the chunk independently)
      partial = asr.transcribe(audio_bytes)
      # return *some* text; UI shows it as live caption
      return JSONResponse({"partial": partial})
  except Exception as e:
      return JSONResponse({"partial": ""})
    
@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    # 1) read uploaded audio
    audio_bytes = await file.read()

    # 2) ASR: audio -> text
    user_text = asr.transcribe(audio_bytes)

    # 3) LLM: text -> reply text (with memory)
    bot_text = llm.generate_response(user_text)

    # 4) TTS: reply text -> wav file path
    audio_path = tts.synthesize_speech(bot_text)

    #test
    print("ASR:", repr(user_text))
    print("LLM:", repr(bot_text))
    from os import path as _p, stat as _s
    print("TTS path:", audio_path, "exists:", _p.exists(audio_path), "size:", (_s(audio_path).st_size if _p.exists(audio_path) else 0))

    # 5) return the wav
    return FileResponse(audio_path, media_type="audio/wav", filename="response.wav")

SENTENCE_SPLIT = re.compile(r'([\.!\?]+[\s"\')\]]+)|(\n{2,})')  # simple-ish

def _split_complete_sentences(buffer: str):
    """Return (sentences, remainder)."""
    out = []
    last = 0
    for m in SENTENCE_SPLIT.finditer(buffer):
        end = m.end()
        chunk = buffer[last:end].strip()
        if chunk:
            out.append(chunk)
        last = end
    rem = buffer[last:].strip()
    return out, rem

@app.post("/chat/stream")
async def chat_stream(file: UploadFile = File(...)):
    # 1) read uploaded audio
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio uploaded")

    # 2) ASR
    user_text = asr.transcribe(audio_bytes)

    # 3) stream LLM deltas → sentence chunks → TTS each → stream WAV parts
    boundary = "wavchunk"
    headers = {"Content-Type": f"multipart/x-mixed-replace; boundary={boundary}"}

    def stream_gen():
        buffer = ""
        # Tell the client the recognized user text first (optional meta header)
        escaped_user = user_text.replace("\\", "\\\\").replace('"', '\\"')
        meta = (
            f'--{boundary}\r\n'
            f'Content-Type: application/json\r\n\r\n'
            f'{{"user":"{escaped_user}"}}\r\n'
        )
        yield meta.encode("utf-8")



        for delta in llm.stream_response(user_text):
            buffer += delta
            sentences, buffer = _split_complete_sentences(buffer)
            for sent in sentences:
                # synthesize a small sentence to wav
                wav_path = tts.synthesize_speech(sent)
                try:
                    with open(wav_path, "rb") as f:
                        data = f.read()
                    part = (
                        f'--{boundary}\r\n'
                        f'Content-Type: audio/wav\r\n'
                        f'Content-Length: {len(data)}\r\n\r\n'
                    ).encode("utf-8") + data + b"\r\n"
                    yield part
                finally:
                    # best-effort cleanup
                    try: os.remove(wav_path)
                    except: pass

        # flush any remainder (last partial sentence)
        if buffer.strip():
            wav_path = tts.synthesize_speech(buffer.strip())
            try:
                with open(wav_path, "rb") as f:
                    data = f.read()
                part = (
                    f'--{boundary}\r\n'
                    f'Content-Type: audio/wav\r\n'
                    f'Content-Length: {len(data)}\r\n\r\n'
                ).encode("utf-8") + data + b"\r\n"
                yield part
            finally:
                try: os.remove(wav_path)
                except: pass

        # end marker
        yield f'--{boundary}--\r\n'.encode("utf-8")

    return StreamingResponse(stream_gen(), headers=headers)