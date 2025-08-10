# Week 3  

- Built a **speech-to-text** pipeline using **Whisper**  
  - **Whisper** – OpenAI speech recognition model for converting audio into text  

- Built a **text-to-text** conversational agent with **OpenAI GPT models**  
  - **OpenAI API** – generated natural language responses with conversational memory  

- Built a **text-to-speech** module using **Coqui**

- Created a **FastAPI** backend to serve the voice agent  
  - **FastAPI** – modern, async Python framework for building APIs  
  - Served endpoints for `/chat/` (audio in → WAV out) and `/history` (chat history)  

- Added **static HTML/JS UI** for voice interaction  
  - **MediaRecorder API** – captured microphone input in the browser  
  - Implemented start/stop recording, file uploads, and real-time partial transcription display  
  - Played agent replies directly in the browser audio player  

- Implemented **conversation memory** and history tracking in backend  
  - Stored alternating user/assistant messages and limited history to recent turns for context  

- Integrated **partial ASR streaming** to send audio chunks while recording  
  - Allowed LLM to start generating responses before recording ended  
  - Planned streaming TTS so replies start playing while still being generated  
