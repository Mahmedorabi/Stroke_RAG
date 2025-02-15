from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from utils.functionlty import bot_func, create_bot_for_selected_bot, extract_pdf_text, analysis_text
from uuid import uuid4
from typing import List, Dict, Generator, Optional
import asyncio
import re

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
    session_id = str(uuid4())
    global uploaded_text
    user_input_with_pdf = f"{chat_input.user_input}\n\nExtracted PDF Text:\n{uploaded_text}" if uploaded_text else chat_input.user_input
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