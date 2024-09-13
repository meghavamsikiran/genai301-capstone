import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

def main():
    """Main function to create the Streamlit interface."""
    st.title("Ed-Tech Q&A")

    # Button to create the knowledge base
    if st.button("Create Knowledgebase"):
        create_vector_db()

    # Input field for the question
    question = st.text_input("Question: ")

    if question:
        # Generate answer using the QA chain
        chain = get_qa_chain()
        if chain:
            try:
                response = chain({"query": question})
                st.header("Answer")
                st.write(response.get("result", "No answer found."))
            except Exception as e:
                st.error(f"An error occurred while getting the answer: {e}")
        else:
            st.error("Failed to create QA chain. Ensure the knowledge base is created.")

if __name__ == "__main__":
    main()
