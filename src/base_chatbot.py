import streamlit as st
from langchain_ollama import ChatOllama
import os
import warnings
from dotenv import load_dotenv


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")


def main():
    st.title("Base chatbot here")
    st.write("Here we create our own chatbot")

    model_name = st.sidebar.selectbox(
        "Select a model",
        ["llama3.2:1b", "deepseekr-1:1b", "mistral:7b", "gemma:2b"],
        index=0,
    )
    with st.form("llm-chatbot"):
        text = st.text_area("Please enter your question")
        submit = st.form_submit_button("Submit")

    def generate_text(input_text, model_nsme):
        model = ChatOllama(model=model_name, base_url="http://localhost:11434/")
        response = model.invoke(input_text)
        return response.content

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if text and model_name and submit:
        with st.spinner("generating response"):
            response = generate_text(text, model_name)
            st.session_state["chat_history"].append({"user": text, "response": response})
            st.write(response)

    st.write(" CHAT HISTORY")
    for chat in reversed(st.session_state["chat_history"]):
        st.write(f"-----USER----: {chat['user']}")
        st.write(f"-----ASSISTANT----: {chat['response']}")
        st.write("----")



if __name__ == "__main__":
    main()
