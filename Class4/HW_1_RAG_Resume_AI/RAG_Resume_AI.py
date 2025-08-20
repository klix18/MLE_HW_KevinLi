# RAG_Resume_AI.py

import os
from dotenv import load_dotenv

# Load environment variables from .env (optional but handy)
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pathlib import Path

def load_dir(path: str):
    all_docs = []
    for p in Path(path).rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == ".pdf":
            all_docs += PyPDFLoader(str(p)).load()
        elif ext in {".txt", ".md"}:
            all_docs += TextLoader(str(p), encoding="utf-8").load()
        elif ext == ".docx":
            all_docs += Docx2txtLoader(str(p)).load()
        # add more types as needed
    return all_docs

docs = load_dir("Documents")


# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# 3) Embed & index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4) RetrievalQA
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

# 5) Ask
print(agent.invoke({"query": "where has kevin worked"})["result"])
