# https://python.langchain.com/v0.2/docs/integrations/chat/ollama/

# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", streaming=True)
# llm = ChatOllama(
#     model="llama3.1",
#     temperature=0.1,
# )

# Document loaders - [https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.csv_loader.CSVLoader.html]

total_extracted_data = []

# CSV Loader
from langchain_community.document_loaders.csv_loader import CSVLoader

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

from langchain_community.document_loaders import PyMuPDFLoader
import os

for file in os.listdir('syllabus_pdfs_csci'):
    loader = PyMuPDFLoader('./syllabus_pdfs_csci/' + file)
    data = loader.load()
    total_extracted_data.append(data)


# Indexing : Split
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
)

all_splits = []
for data in total_extracted_data:
    split = text_splitter.split_documents(data)
    all_splits.extend(split)


# print(len(all_splits))


# Indexing : Store - Embed the contents of each document split and insert these embeddings into a vector database (or vector store)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(api_key = os.environ.get("OPENAI_API_KEY")))

from langchain.prompts import ChatPromptTemplate

retriever = vectorstore.as_retriever()
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


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
#

# print(retriever | format_docs)
runnable = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# result = runnable.invoke("which courses with name and course code, should I take for machine learning?")
# print(result)

import asyncio
# Function to handle streaming results
async def stream_results(question):
    for chunk in runnable.stream(question):
        print(chunk, end="")  # or process the chunk as needed

# Example usage
if __name__ == "__main__":
    question = "which courses with name and course code, should I take for machine learning?"
    asyncio.run(stream_results(question))
