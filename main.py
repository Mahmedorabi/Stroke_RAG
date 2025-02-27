from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional, Generator
import asyncio
import re
import uuid
import sounddevice as sd
import wavio
import speech_recognition as sr
import os
from pathlib import Path
from utils.functionlty import *

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the bot
bot = create_bot_for_selected_bot(
    embeddings="BAAI/bge-base-en-v1.5",
    vdb_dir="Stroke_vdb",
    sys_prompt_dir="assist/prompt.txt",
    name="stroke RAG"
)

chat_history: List[Dict[str, str]] = []
uploaded_text: Optional[str] = None  

class ChatInput(BaseModel):
    user_input: str

class VoiceInput(BaseModel):
    input_language: str

async def stream_chat_response(bot, user_input: str, session_id: str) -> Generator[str, None, None]:
    response_buffer = [] 
    for chunk in bot_func(bot, user_input, session_id=session_id):
        if isinstance(chunk, str):  
            # Format camel case and punctuation
            formatted_chunk = re.sub(r'([A-Z][a-z]+)', r' \1', chunk)  # Split camel case
            formatted_chunk = re.sub(r'([.!?])', r'\1 ', formatted_chunk)  # Add space after punctuation
            response_buffer.append(formatted_chunk)
            yield f"data: {formatted_chunk}\n\n"
        elif isinstance(chunk, dict):  
            if answer_chunk := chunk.get("answer"):
                formatted_chunk = re.sub(r'([A-Z][a-z]+)', r' \1', answer_chunk)
                formatted_chunk = re.sub(r'([.!?])', r'\1 ', formatted_chunk)
                response_buffer.append(formatted_chunk)
                yield f"data: {formatted_chunk}\n\n"
        await asyncio.sleep(0.1)  
    
    full_response = "".join(response_buffer)
    chat_history.append({"sender": "assistant", "message": full_response})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    global uploaded_text
    try:
        text = extract_pdf_text(file.file)
        uploaded_text = analysis_text(text)
        return {
            "status": "success",
            "message": "PDF uploaded successfully!",
            "analysis": uploaded_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/chat")
async def chat(chat_input: ChatInput):
    session_id = str(uuid.uuid4())
    global uploaded_text
    user_input_with_pdf = f"{chat_input.user_input}\n\nExtracted PDF Text:\n{uploaded_text}" if uploaded_text else chat_input.user_input
    uploaded_text = None  
    chat_history.append({"sender": "user", "message": user_input_with_pdf})
    return StreamingResponse(
        stream_chat_response(bot, user_input_with_pdf, session_id),
        media_type="text/event-stream"
    )

@app.post("/voice_chat")
async def voice_chat(voice_input: VoiceInput):
    session_id = str(uuid.uuid4())
    global uploaded_text
    
    # Process voice input
    transcribed_text = process_voice_input(voice_input.input_language)
    if transcribed_text.startswith("Error"):
        raise HTTPException(status_code=500, detail=transcribed_text)
    
    user_input_with_pdf = f"{transcribed_text}\n\nExtracted PDF Text:\n{uploaded_text}" if uploaded_text else transcribed_text
    uploaded_text = None  
    chat_history.append({"sender": "user", "message": user_input_with_pdf})
    return StreamingResponse(
        stream_chat_response(bot, user_input_with_pdf, session_id),
        media_type="text/event-stream"
    )

@app.get("/chat_history")
async def get_chat_history():
    return {"chat_history": chat_history}

@app.get("/")
async def get_ui():
    return HTMLResponse(content=open("static/index.html").read())

def record_audio(duration=5, sample_rate=44100):
    try:
        recording = sd.rec(int(duration * sample_rate),
                           samplerate=sample_rate,
                           channels=1,
                           dtype='int16')
        sd.wait()
        # Save the file
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file = temp_dir / "temp_recording.wav"
        wavio.write(str(temp_file), recording, sample_rate, sampwidth=2)
        
        return str(temp_file)
    except Exception as e:
        return f"Error recording audio: {str(e)}"
    
from pydub import AudioSegment
import os

def convert_to_wav(audio_file):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(audio_file)
        # Convert to WAV format
        wav_file = audio_file.replace(".webm", ".wav")
        audio.export(wav_file, format="wav")
        return wav_file
    except Exception as e:
        raise Exception(f"Error converting audio to WAV: {str(e)}")

@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile = File(...), input_language: str = Form(...)):
    try:
        # Save the uploaded file
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / "uploaded_audio.webm"  # Assume the file is in webm format
        with open(temp_file, "wb") as buffer:
            buffer.write(file.file.read())
        
        # Convert to WAV format
        wav_file = convert_to_wav(str(temp_file))
        
        # Transcribe the audio
        text = transcribe_audio_file(wav_file, input_language)
        return {"transcribed_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(wav_file):
            os.remove(wav_file)

def transcribe_audio_file(audio_file, input_language):
    try:
        # Map language selection to Google Speech Recognition language codes
        language_codes = {
            "Arabic": "ar-AR",
            "English": "en-US"
        }
        language_code = language_codes.get(input_language, "en-US")
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language=language_code)
            return text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def process_voice_input(input_language):
    try:
        audio_file = record_audio()
        if audio_file.startswith("Error"):
            return audio_file
        text = transcribe_audio(audio_file, input_language)
        return text
    except Exception as e:
        return f"Error processing voice input: {str(e)}"