from utils.functionlty import analysis_text, extract_pdf_text, bot_func, create_bot_for_selected_bot, process_voice_input
import streamlit as st
from uuid import uuid4
import base64
import os
from pathlib import Path

# Initialize the bot
bot = create_bot_for_selected_bot(
    embeddings="BAAI/bge-base-en-v1.5",
    vdb_dir="Stroke_vdb",
    sys_prompt_dir="assist/prompt.txt",
    name="stroke RAG"
)

# Configure logo
logo_path = "assist/download-removebg-preview.png"
if not os.path.exists(logo_path):
    st.error(f"Logo file not found at: {logo_path}")
    st.stop()

@st.cache_resource
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

encoded_logo = get_base64_encoded_image(logo_path)

# Main UI Layout
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_logo}" alt="Logo" style="width:70px; margin-right: 10px;">
        <h1>Stroke Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    input_language = st.selectbox("Select Input Language", ["English", "Arabic"])
    uploaded_file = st.file_uploader("Upload PDF Report", type="pdf")

# Chat History Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat History
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)

# Voice Input Section
col1, col2 = st.columns([0.85, 0.15])
with col1:
    user_input = st.chat_input(placeholder="Type or speak your message...")
with col2:
    if st.button("🎤", help="Record voice input (5 seconds)", key="voice_button"):
        with st.spinner("Recording..."):
            transcribed_text = process_voice_input(input_language)
            if transcribed_text and not transcribed_text.startswith("Error"):
                st.session_state.voice_input = transcribed_text
            else:
                st.error(transcribed_text)

# Handle Voice Input
if "voice_input" in st.session_state:
    user_input = st.session_state.voice_input
    del st.session_state.voice_input

# Process PDF if uploaded
final_text = ""
if uploaded_file is not None:
    text = extract_pdf_text(uploaded_file)
    final_text = analysis_text(text)

# Combine PDF analysis with user input
if user_input:
    if final_text:
        user_input += f"\n{final_text}"
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input.replace(final_text, "").strip())

    # Generate and display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        for response in bot_func(bot, user_input, session_id=str(uuid4())):
            full_response += response
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)

    # Update chat history
    st.session_state.chat_history.append(("You", user_input.replace(final_text, "").strip()))
    st.session_state.chat_history.append(("assistant", full_response))

