from langchain_community.vectorstores import FAISS
from langchain_community.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Use the provided Google API key directly
llm = GooglePalm(google_api_key='AIzaSyDYhCjDdFZNRl2FPWF8zwAUP1D1eOSFMOY', temperature=0.1)

# Use SentenceTransformer for embeddings
embedding_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

vectordb_file_path = "faiss_index"

def create_vector_db():
    """Creates and saves the vector database from the CSV file."""
    # Load FAQ data from CSV
    loader = CSVLoader(file_path='ed-tech_faqs.csv', source_column="prompt")
    data = loader.load()

    # Generate embeddings
    texts = [doc.page_content for doc in data]
    embeddings = embedding_model.encode(texts)

    # FAISS expects the embeddings to be a list of tuples [(text1, embedding1), (text2, embedding2), ...]
    text_embeddings = list(zip(texts, embeddings))

    # Create FAISS vector database
    vectordb = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=embedding_model)

    # Save vector database
    vectordb.save_local(vectordb_file_path)
    print("Knowledge base created and saved.")

def get_qa_chain():
    """Loads the QA chain with the vector database and returns it."""
    try:
        # Load the vector database with dangerous deserialization allowed
        vectordb = FAISS.load_local(vectordb_file_path, embedding_model.encode, allow_dangerous_deserialization=True)

        # Create a retriever
        retriever = vectordb.as_retriever(score_threshold=0.7)

        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer, try to provide as much text as possible from the "response" section in the source document context without making many changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Create a RetrievalQA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever,
            input_key="query", return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        return chain

    except Exception as e:
        print(f"An error occurred while creating the QA chain: {e}")
        return None
