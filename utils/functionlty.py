# __import__("pysqlite3")
# import sys

# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import PyPDF2
from langchain_together import ChatTogether
from langchain_core.runnables.history import RunnableWithMessageHistory
import speech_recognition as sr
import sounddevice as sd
import wavio
import os
from pathlib import Path



msgs = StreamlitChatMessageHistory(key="special_app_key")
llm = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo",temperature=0.0,api_key="3dfa2e31d4d2d1e7c7751a34ccad57a494bd4bca8e045164832cd900a75f49ba")



def read_db(filepath: str, embeddings_name):
    embeddings = HuggingFaceBgeEmbeddings(model_name=embeddings_name)
    vectordb = Chroma(persist_directory=filepath, embedding_function=embeddings)
    retreiver = vectordb.as_retriever()
    return retreiver


def read_system_prompt(filepath: str):

    with open(filepath, "r") as file:
        prompt_content = file.read()

    context = "{context}"

    system_prompt = f'("""\n{prompt_content.strip()}\n"""\n"{context}")'

    return system_prompt


def create_conversational_rag_chain(sys_prompt_dir, vdb_dir, llm, embeddings_name):
    retriever = read_db(vdb_dir, embeddings_name)

    contextualize_q_system_prompt = """Given a chat history and the latest user question
    which might reference context in the chat history,
    formulate a response which can be understood and clear
    without the chat history. Do NOT answer the question,
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    sys_prompt = read_system_prompt(sys_prompt_dir)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def extract_pdf_text(file_object):
    reader = PyPDF2.PdfReader(file_object)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return str(text)



def analysis_text(text: str):

    detailed_prompt = f"""
    You are a medical report analysis assistant. Your task is to:
    
    1. Carefully review the medical report text
    2. Identify key medical values and their significance
    3. Provide a clear, concise interpretation in a user-friendly format
    4. Follow this output structure strictly:
    
    Medical Value Interpretation Guide:
    - If a value is outside normal range, indicate:
      * The specific value
      * Whether it's high or low
      * Potential implications (in simple language)
    - Use clear, non-technical language
    - Avoid medical jargon
    - Provide actionable insights
    
    Example Output Format:
    ```
    Glucose Level: 228.69 mg/dL
    ⚠️ Status: High
    Interpretation: Your glucose level is elevated, which may indicate:
    - Potential pre-diabetic condition
    - Need for dietary adjustments
    - Recommend consulting your healthcare provider

    Recommendation: Schedule a follow-up blood test
    ```

    Report Text:
    {text}

    Your Analysis:
    """
    return detailed_prompt


# def bot_func(rag_chain, user_input, session_id):
#     # Stream the response from the rag_chain
#     for chunk in rag_chain.stream(
#         {"input": user_input}, config={"configurable": {"session_id": session_id}}
#     ):
#         if isinstance(chunk, str):  # Handle string responses
#             yield {"answer": chunk}
#         elif isinstance(chunk, dict):  # Handle dictionary responses
#             if answer_chunk := chunk.get("answer"):
#                 yield {"answer": answer_chunk}

def bot_func(rag_chain, user_input, session_id):
    for chunk in rag_chain.stream(
        {"input": user_input}, config={"configurable": {"session_id": session_id}}
    ):
        if answer_chunk := chunk.get("answer"):
            yield answer_chunk    

def create_bot_for_selected_bot(name, embeddings, vdb_dir, sys_prompt_dir):
    """Create a bot for the selected configuration."""
    rag_chain = create_conversational_rag_chain(
        sys_prompt_dir, vdb_dir, llm, embeddings
    )
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,  
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
        max_tokens_limit=500,
        top_n=5
    )
    return conversational_rag_chain


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
    
def transcribe_audio(audio_file, input_language):
    try:
        # Map language selection to Google Speech Recognition language codes
        language_codes = {
            "Arabic": "ar-AR",
            "English": "en-US"
        }
        language_code = language_codes.get(input_language, "en-US")
        
        recognizer = sr.Recognizer()
        # open a audio file
        with sr.AudioFile(audio_file) as source:
            # Read audio content
            audio = recognizer.record(source)
            
            text = recognizer.recognize_google(audio, language=language_code)
            return text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)

def process_voice_input(input_language):
    try:
        audio_file = record_audio()
        if audio_file.startswith("Error"):
            return audio_file
        text = transcribe_audio(audio_file, input_language)
        return text
    except Exception as e:
        return f"Error processing voice input: {str(e)}"
