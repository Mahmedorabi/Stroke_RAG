from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from typing import Dict, List, Optional, Any
from utils.functionlty import create_bot_for_selected_bot, bot_func, extract_pdf_text, analysis_text

app = FastAPI(title="RAG Chat Application")

class ChatRequest(BaseModel):
    input: str
    session_id: str

# Store bots, chat histories, and uploaded texts for different sessions
active_bots: Dict[str, Any] = {}
chat_histories: Dict[str, List[Dict[str, str]]] = {}
uploaded_texts: Dict[str, Optional[str]] = {}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming endpoint for chat responses"""
    if not request.session_id or not request.input:
        raise HTTPException(status_code=400, detail="Session ID and input are required")

    # Get or create chat history for this session
    session_chat_history = chat_histories.setdefault(request.session_id, [])

    # Get uploaded text for this session and remove it after use
    uploaded_text = uploaded_texts.pop(request.session_id, None)
    user_input_with_pdf = f"{request.input}\n\nExtracted PDF Text:\n{uploaded_text}" if uploaded_text else request.input

    # Append user's message to chat history
    session_chat_history.append({"sender": "user", "message": user_input_with_pdf})

    # Get or create bot for this session
    if request.session_id not in active_bots:
        embeddings = "BAAI/bge-base-en-v1.5"
        vdb_dir = "Stroke_vdb"
        sys_prompt_dir = "assist/prompt.txt"
        bot = create_bot_for_selected_bot("default", embeddings, vdb_dir, sys_prompt_dir)
        active_bots[request.session_id] = bot
    else:
        bot = active_bots[request.session_id]

    async def stream_response():
        try:
            response_chunks = []
            for chunk in bot_func(bot, user_input_with_pdf, request.session_id):
                response_chunks.append(chunk)
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.01)  # Small delay for smooth streaming
            full_response = "".join(response_chunks)
            session_chat_history.append({"sender": "assistant", "message": full_response})
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/pdf/upload")
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    """Upload a PDF file and store its analyzed text for the session"""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        text = extract_pdf_text(file.file)
        analysis = analysis_text(text)
        uploaded_texts[session_id] = analysis
        return {
            "status": "success",
            "message": "PDF Uploaded successfully",
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/chat_history")
async def get_chat_history(session_id: str = Form(...)):
    """Retrieve chat history for a specific session"""
    if session_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"chat_history": chat_histories[session_id]}

