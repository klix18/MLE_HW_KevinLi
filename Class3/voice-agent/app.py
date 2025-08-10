from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import asr
import llm
import tts
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import StreamingResponse



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
