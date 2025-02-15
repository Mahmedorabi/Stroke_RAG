from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from utils.functionlty import analysis_text, extract_pdf_text, bot_func, create_bot_for_selected_bot
from uuid import uuid4
from typing import List, Dict
import uvicorn


app = FastAPI()

# Initialize the bot
bot = create_bot_for_selected_bot(
    embeddings="BAAI/bge-base-en-v1.5",
    vdb_dir="Stroke_vdb",
    sys_prompt_dir="assist/prompt.txt",
    name="stroke RAG"
)

# In-memory storage for chat history (for demonstration purposes)
chat_history: List[Dict[str, str]] = []


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    text = extract_pdf_text(file.file)
    final_text = analysis_text(text)
    return {"analysis": final_text}

from pydantic import BaseModel

class ChatInput(BaseModel):
    user_input: str

@app.post("/chat")
async def chat(chat_input: ChatInput):
    session_id = str(uuid4())
    response_chunks = []
    for chunk in bot_func(bot, chat_input.user_input, session_id=session_id):
        response_chunks.append(chunk.get("answer", ""))
    response = "".join(response_chunks)
    
    # Store the chat history
    chat_history.append({"sender": "user", "message": chat_input.user_input})
    chat_history.append({"sender": "assistant", "message": response})
    
    return {"response": response, "chat_history": chat_history}

@app.get("/chat_history")
async def get_chat_history():
    return {"chat_history": chat_history}

