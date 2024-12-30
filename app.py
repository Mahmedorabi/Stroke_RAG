from utils.functionlty import analysis_text, extract_pdf_text, bot_func, create_bot_for_selected_bot
import streamlit as st
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_together import ChatTogether
from uuid import uuid4
import PyPDF2
import base64
import os

bot = create_bot_for_selected_bot(
    embeddings="BAAI/bge-base-en-v1.5",
    vdb_dir="Stroke_vdb",
    sys_prompt_dir="assist/prompt.txt",
    name = "stroke RAG"
)


logo_path = "assist/download-removebg-preview.png"
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