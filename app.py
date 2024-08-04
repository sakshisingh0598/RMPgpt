from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

import os


import chainlit as cl
# https://docs.chainlit.io/get-started/overview
from dotenv import load_dotenv

load_dotenv()

@cl.on_chat_start
async def on_chat_start():

    total_extracted_data = []

    # CSV Loader
    loader = CSVLoader(
        file_path = './csci-20241.csv',
        csv_args={
            'delimiter': ',',

            # "Course number","Course title","Registration restrictions","Units","Type","Section","Session","Time","Days","Seats","Registered","Waitlist","Instructor","Room"
            # 'fieldnames': ['Index', 'Height', 'Weight']
        }
    )
    data = loader.load()
    total_extracted_data.append(data)


    # PDF Loader - PyMuPDF (other also exist) - [https://python.langchain.com/v0.2/docs/how_to/document_loader_pdf/]

    for file in os.listdir('syllabus_pdfs_csci'):
        loader = PyMuPDFLoader('./syllabus_pdfs_csci/' + file)
        data = loader.load()
        total_extracted_data.append(data)


    # Indexing : Split

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )

    all_splits = []
    for data in total_extracted_data:
        split = text_splitter.split_documents(data)
        all_splits.extend(split)


    # print(len(all_splits))


    # Indexing : Store - Embed the contents of each document split and insert these embeddings into a vector database (or vector store)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(api_key = os.environ.get("OPENAI_API_KEY")))

    retriever = vectorstore.as_retriever()


    # Chain
    model = ChatOpenAI(model="gpt-4o", streaming=True)

    system_prompt = (
        "You are a smart AI assistant which help students to take courses they are insterested. You have been provided with all the courses list details and syllabus university offers. Use only these documents and help students."
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    runnable = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | StrOutputParser())
    # runnable = ({"context": retriever | format_docs} | prompt | model | StrOutputParser())
    # runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
