from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from utils.functionlty import bot_func, create_bot_for_selected_bot, extract_pdf_text, analysis_text
from uuid import uuid4
from typing import List, Dict, Generator, Optional
import asyncio

app = FastAPI()

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
            response_buffer.append(chunk)
            yield f"data: {chunk}\n\n"
        elif isinstance(chunk, dict):  
            if answer_chunk := chunk.get("answer"):
                response_buffer.append(answer_chunk)
                yield f"data: {answer_chunk}\n\n"
        await asyncio.sleep(0.1)  
    
    full_response = "".join(response_buffer)
    chat_history.append({"sender": "assistant", "message": full_response})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    global uploaded_text
    text = extract_pdf_text(file.file)
    uploaded_text = analysis_text(text)  
    return {"analysis": uploaded_text}

@app.post("/chat")
async def chat(chat_input: ChatInput):
    session_id = str(uuid4())
    global uploaded_text

    if uploaded_text:
        user_input_with_pdf = f"{chat_input.user_input}\n\nExtracted PDF Text:\n{uploaded_text}"
        uploaded_text = None  
    else:
        user_input_with_pdf = chat_input.user_input
    
    chat_history.append({"sender": "user", "message": user_input_with_pdf})
    
    return StreamingResponse(
        stream_chat_response(bot, user_input_with_pdf, session_id),
        media_type="text/event-stream",  
    )

@app.get("/chat_history")
async def get_chat_history():
    return {"chat_history": chat_history}