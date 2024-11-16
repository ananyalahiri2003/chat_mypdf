"""
Before getting started:
Make sure
1. Langsmith - https://docs.smith.langchain.com/old/tracing/faq/logging_and_viewing We create an account free of cost - and copy the LANGCHAIN_API_KEY.

2. In a .env file -
LANGCHAIN_API_KEY - Set
LANGCHAIN_PROJECT="chat_mypdf"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com/"
LANGCHAIN_TRACING_V2 = True

3. Install Ollama https://ollama.com/download, then double click and open Ollama application, click and install command line from application.
https://ollama.com/library/llama3.2:1b, I see the command line to run it. I copy the command line below, and open terminal and run -
ollama run llama3.2:1b (It says pulling manifest).
Then ask "hello", it responds in next command line. To exit type /bye.
Next time go to the dir and type ollama run llama3.2:1b to bring up the same Q&A prompt.

4. Nomic-embed-text - https://ollama.com/library/nomic-embed-text
On terminal
ollama pull nomic-embed-text

Note Ollama runs on http://localhost:11434

"""

import os
import warnings
import faiss
import tiktoken
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain import hub      # From hub we will pull out RAG-related prompts
from langchain_core.output_parsers import StrOutputParser      # Using StrOutputParser we will get final output as string data
from langchain_core.runnables import RunnableMap, RunnablePassthrough      # We can pass through the question and the context directly to our LLM
from langchain_core.prompts import ChatPromptTemplate      # We need to pass the prompt inside the ChatPromptTemplate and thereafter question and context chunks

from langchain_ollama import ChatOllama      # ChatOllama helps connect Llama3.2 running on Ollama server, and Langchain framework


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
root_dir = '/Users/ananyalahiri/PycharmProjects/chat_myPDF/src/rag-dataset'  # Please substitute with your own dir loc


def main():
    load_dotenv()
    api_key = os.environ['LANGCHAIN_API_KEY']

    # loader = PyMuPDFLoader('/Users/ananyalahiri/PycharmProjects/chat_myPDF/src/rag-dataset/gym_supplements/1. Analysis of Actual Fitness Supplement.pdf')
    # docs = loader.load()

    pdfs = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pdf'):
                pdfs.append(os.path.join(root, file))
    print(f"{len(pdfs)=}")

    docs = []
    for pdf in pdfs:
        loader = PyMuPDFLoader(pdf)
        pages = loader.load()
        docs.extend(pages)
    print(f"{len(docs)=}")

    # Now we use recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    chunks = text_splitter.split_documents(docs)
    print(f"{len(docs)=}, {len(chunks)=}")

    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    print(f"{len(encoding.encode(docs[0].page_content))=}, {len(encoding.encode(chunks[0].page_content))=}")

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    # If we see http://localhost:11434 we see Ollama is running
    single_vector = embeddings.embed_query("this is some text data")
    print(f"{len(single_vector)=}")

    index = faiss.IndexFlatL2(len(single_vector))
    print(f"{index.ntotal=}, {index.d=}")

    vector_store = FAISS(
        embedding_function=embeddings,  # Ollama nomic-embed-text
        index=index,  # The index we calced
        docstore=InMemoryDocstore(),  # We will store the documents in memory
        index_to_docstore_id={},  # An empty dictionary as of now, once we add document to vectorstoe we will have
        # all document IDs inside this vectorstore
    )
    print(f"{vector_store=}")

    # We had already chunked the docs into 321 chunks.
    ids = vector_store.add_documents(documents=chunks[:5])
    print(f"{len(ids)=}")
    print(f"{vector_store.index_to_docstore_id=}")

    # # Let us store in index.faiss and index.pkl and test
    # db_name = "health_supplements_store"
    # vector_store.save_local(db_name)
    #
    # # Let us load
    #
    # new_vector_store = FAISS.load_local(
    #     db_name,
    #     embeddings=embeddings,
    #     allow_dangerous_deserialization=True,
    # )  # We need to provide same embeddings

    # Retrieval
    question = "what is used to gain muscle mass?"
    docs = vector_store.search(query=question, search_type="similarity")

    print("For raw vector_store search")
    for doc in docs:
        print(doc.page_content)
        print("\n")

    # Convert to retriever
    print("Let us try retriever")
    retriever = vector_store.as_retriever(search_type="mmr",
                                          search_kwargs={
                                              'k': 3,
                                              'fetch_k': 100,
                                              'lambda_mult': 1,    # lambda_mult means only relevant to query}
                                          })
    docs = retriever.invoke(question)
    for doc in docs:
        print(doc.page_content)
        print("\n")

    # Now we should be able to see these questions in Langsmith api - go to https://smith.langchain.com/o/d6e8507a-7896-4388-ad97-63a486159534/projects?paginationState=%7B%22pageIndex%22%3A0%2C%22pageSize%22%3A10%7D
    # Then click chat_pdf project and drill down into documentd

    question = "What are the benefits of BCAA supplement?"
    docs = retriever.invoke(question)
    print(f"{question=}")
    for doc in docs:
        print(doc.page_content)
        print("\n")

    # Now we will not query from vector_store - we will pass everything into Llama2 model
    # RAG with Llama3.2 on Ollama
    model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")
    model.invoke("hi")
    # We go to Langchain https://smith.langchain.com/hub?organizationId=d6e8507a-7896-4388-ad97-63a486159534
    # Look for prompts RAG
    # We can drill down to RAG prompt within https://smith.langchain.com/hub/jisujiji/rag-prompt-1?organizationId=d6e8507a-7896-4388-ad97-63a486159534
    # You can also see ChatPromptTempplate towards bottom of each page
    # We use this one https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=d6e8507a-7896-4388-ad97-63a486159534
    prompt = hub.pull("rlm/rag-prompt")
    print(f"{prompt=}")

    # We can also use our own prompt - do modification to an existing prompt. We copy paste the prompt, and include
    # extra condition.
    prompt = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        If possible answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only. 
        Question: {question}
        Context: {context}
        Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    print(f"Custom {prompt=}")

    # CREATE RAG chain
    # Let us combine chunk data into one document - we can combine the documents
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    print(f"format_docs: {format_docs(docs)}")

    # RAG chain - we need runnable
    # Whenever we pass in a document into the retriever, retriever.invoke will
    rag_chain = (
            RunnableMap(
                {"context": retriever|format_docs, # Whenever we pass a question through retriever.invoke, it will return set of documents. Those will pass automatically into format_docs, and get stored unde3r context dictionary
                "question": RunnablePassthrough()  # We create an empty RunnablePassThrough,
                 },
            )
            | prompt  # Pass context and question into the prompt template
            | model  # Usw the model to generate response
            | StrOutputParser()  # Parse the model's output into a string
    )

    question = "what is used to gain muscle mass?"

    output = rag_chain.invoke(question)
    print(f"Question: {question}")
    print("\n")
    print(f"{output=}")

    # We can see the output in Langsmith
    





if __name__ == "__main__":
    main()