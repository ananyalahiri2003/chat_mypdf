import streamlit as st
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, Request
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)

from src.chat_bot_history import chat_history


class ChatRequest(BaseModel):
    user_input: str
    chat_history: list


app = FastAPI()

st.title("FastAPI App")
st.write("Create a FastAPI endpoint for a streamlit chatbot")

model_name = st.sidebar.selectbox(
    "Select a model",
    ["llama3.2:1b", "mistral:7b", "gemma:2b", "deepseekr-1:1b"],
    index=0
)

system_message = SystemMessagePromptTemplate.from_template(
    f"You are a helpful AI assistant. You are playing the role of a very intelligent mother, who answers questions posed by Ahana, a super intelligent 18 year old mathematician girl. Answer truthfully, and think step by step."
)

with st.form("llm-chatbot"):
    text = st.text_area("Please enter your question here")
    submit = st.form_submit_button("Submit")

model = ChatOllama(model=model_name, base_url="http://localhost:11434/")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

@app.post("/generate")
def generate_response(request: ChatRequest):
    prompt = HumanMessagePromptTemplate.from_template(request.user_input)
    chat_history = [system_message]
    for chat in request.chat_history:
        chat_history.append(HumanMessagePromptTemplate.from_template(chat['user']))
        chat_history.append(AIMessagePromptTemplate.from_template(chat['assistant']))

    chat_history.append(prompt)

    chat_template = ChatPromptTemplate.from_messages(chat_history)
    chain = chat_template | model | StrOutputParser()
    response = chain.invoke({})
    return response

if text and submit:
    with st.spinner("Generating response):"):
        user_prompt = HumanMessagePromptTemplate.from_template(text)
        chat_history = st.session_state["chat_history"]
        chat_request = ChatRequest(user_input=user_prompt, chat_history=chat_history)
        response = generate_response(chat_request)
        st.session_state["chat_history"].append(f"user: {text}, assistant: {response}")

for chat in st.session_state["chat_history"]:
    st.write(f"User: {chat['user']}")
    st.write(f"Assistant: {chat['assistant']}")
    st.write("---")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
