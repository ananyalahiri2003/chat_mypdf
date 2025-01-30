import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

st.title(":brain: Make your own chat history enabled chatbot")
st.write("Let's go!!")

model_name = st.sidebar.selectbox(
    "Select a model",
    ["llama3.2:1b", "deepseekr-1:1b", "mistral:7b", "gemma:2b"],
    index=0,
)
model = ChatOllama(model=model_name, base_url = "http://localhost:11434/")

system_message = SystemMessagePromptTemplate.from_template("You are a helpful AI assistant. You are mummy to Ahana. You explain things like a mother to her teenaged intelligent daughter")

with st.form("llm-chat-history"):
    text = st.text_area("Please enter your question here")
    submit = st.form_submit_button("Submit")

def generate_response(chat_history):
    chat_template = ChatPromptTemplate.from_messages(chat_history)
    chain = chat_template | model | StrOutputParser()
    response = chain.invoke({})
    return response

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def get_history():
    chat_history = [system_message]
    for chat in st.session_state["chat_history"]:
        prompt = HumanMessagePromptTemplate.from_template(chat['user'])
        chat_history.append(prompt)

        ai_message = AIMessagePromptTemplate.from_template(chat['assistant'])
        chat_history.append(ai_message)

    return chat_history

if submit and model and text:
    with st.spinner("Generating response"):
        prompt = HumanMessagePromptTemplate.from_template(text)
        chat_history = get_history()
        chat_history.append(prompt)

        response = generate_response(chat_history)
        st.session_state['chat_history'].append({'user': text, 'assistant': response})

st.write("Chat history")
for chat in st.session_state["chat_history"]:
    st.write(f"User: {chat['user']}")
    st.write(f"Assistant: {chat['assistant']}")
    st.write("----")

