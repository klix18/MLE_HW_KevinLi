import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from IPython.display import display, HTML

class RAGClass:
    def __init__(self, data_path: str):
        """
        Initialize the RAGClass with the path to the data file.
        """
        self.data_path = data_path
        self.documents = []
        self.text_chunks = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def load_documents(self):
        """
        Loads documents from the specified data path and stores them in self.documents.
        Returns the loaded documents.
        """
        loader = TextLoader(self.data_path)
        self.documents = loader.load()
        # print(f"Loaded {len(self.documents)} documents.")
        for i, doc in enumerate(self.documents):
            preview = doc.page_content[:200].replace('\n', '<br>')
            display(HTML(f"<b>Document {i+1} content preview:</b><br>{preview}{'...' if len(doc.page_content) > 200 else ''}<hr>"))
        return self.documents

    def split_documents(self, chunk_size=500, chunk_overlap=50):
        """
        Splits loaded documents into smaller chunks for processing.
        Returns the list of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"Split documents into {len(self.text_chunks)} chunks.")
        for i, chunk in enumerate(self.text_chunks):
            formatted_text = chunk.page_content.replace('. ', '.<br>')
            display(HTML(f"<b>Chunk {i+1}:</b><br>{formatted_text}<hr>"))
        return self.text_chunks

    def create_vectorstore(self):
        """
        Creates a vector store from the split text chunks using OpenAI embeddings.
        Returns the vectorstore object.
        """
        if not self.text_chunks:
            raise ValueError("No text chunks found. Please split documents before creating the vector store.")
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(self.text_chunks, embedding=embeddings)
        print("Vectorstore created with embedded documents.")
        display(HTML("<b>Vectorstore Contents:</b>"))
        for i, doc in enumerate(self.text_chunks):
            formatted_text = doc.page_content.replace('. ', '.<br>')
            display(HTML(f"<b>Document {i+1}:</b><br>{formatted_text}<hr>"))
        return self.vectorstore

    def setup_retriever(self):
        """
        Sets up a retriever from the vectorstore for similarity search.
        Returns the retriever object.
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        self.retriever = self.vectorstore.as_retriever()
        print("Retriever set up from vectorstore.")
        display(HTML(f"<b>Retriever details:</b> {self.retriever}"))
        return self.retriever

    def setup_qa_chain(self):
        """
        Initializes the QA chain using a language model and the retriever.
        Returns the QA chain object.
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized.")
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=self.retriever)
        print("QA chain initialized with LLM and retriever.")
        display(HTML(f"<b>QA chain details:</b> {self.qa_chain}"))
        return self.qa_chain

    def answer_query(self, query: str):
        """
        Answers a query using the QA chain.
        Returns the answer string.
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")
        result = self.qa_chain.run(query)
        display(HTML(f"<b>Query:</b> {query}<br><b>Answer:</b> {result}"))
        return result

    def evaluate(self, queries: list, ground_truths: list):
        """
        Evaluates the QA system using a list of queries and ground truths.
        Returns the accuracy as a float.
        """
        if len(queries) != len(ground_truths):
            raise ValueError("Queries and ground truths must be of the same length.")
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")
        correct = 0
        for idx, (query, truth) in enumerate(zip(queries, ground_truths)):
            answer = self.qa_chain.run(query)
            display(HTML(f"<b>Query {idx+1}:</b> {query}<br><b>Expected:</b> {truth}<br><b>Model Answer:</b> {answer}<hr>"))
            if truth.lower() in answer.lower():
                correct += 1
        accuracy = correct / len(queries)
        display(HTML(f"<b>Evaluation Accuracy:</b> {accuracy * 100:.2f}%"))
        return accuracy