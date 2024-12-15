__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


msgs = StreamlitChatMessageHistory(key="special_app_key")



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
