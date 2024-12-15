from utils.functionlty import create_conversational_rag_chain
import streamlit as st
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_together import ChatTogether
from uuid import uuid4
import PyPDF2
import base64
import os
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

llm = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo",temperature=0.0,api_key="1b4b0c2624f2a3f595a50d4da9424898a53a23f824fda9b1651dfa895edfa2ac")

msgs = StreamlitChatMessageHistory(key = "special_app_key")


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

def bot_func(rag_chain, user_input, session_id):
    for chunk in rag_chain.stream(
        {"input": user_input}, config={"configurable": {"session_id": session_id}}
    ):
        if answer_chunk := chunk.get("answer"):
            yield answer_chunk

@st.cache_resource
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


bot = create_bot_for_selected_bot(
    embeddings="BAAI/bge-base-en-v1.5",
    vdb_dir="/home/mohammedorabi2002/Graduation_projects/RAG stroke/Stroke_vdb",
    sys_prompt_dir="/home/mohammedorabi2002/Graduation_projects/RAG stroke/assist/prompt.txt",
    name = "stroke RAG"
)
logo_path = "/home/mohammedorabi2002/Graduation_projects/RAG stroke/assist/download-removebg-preview.png"
if not os.path.exists(logo_path):
    st.error(f"Logo file not found at: {logo_path}")
    st.stop()


@st.cache_resource
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

encoded_logo = get_base64_encoded_image(logo_path)
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_logo}" alt="Logo" style="width:70px; margin-right: 10px;">
        <h1>Stroke Chatbots </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
uploaded_file = st.sidebar.file_uploader("Choose your .pdf file", type="pdf")
user_input = st.chat_input(placeholder="Your message")
user_input_2 = user_input

if uploaded_file is not None:
    text = analysis_text(uploaded_file)
    if user_input:
        user_input += text

for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "you" else "assistant"):
        st.markdown(message)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input_2)

    with st.chat_message("assistant"):
        response_placholder = st.empty()
        full_response = ""
        for response in bot_func(bot,user_input,session_id=str(uuid4())):
            full_response += response
            response_placholder.markdown(full_response)

    
    st.session_state.chat_history.append(("You", user_input_2))
    st.session_state.chat_history.append(("assistant", full_response))