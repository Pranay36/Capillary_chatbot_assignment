import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Load .env with your Google API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Build QA Chain
def get_conversational_chain():
    prompt_template = """
You are an AI assistant that provides detailed explanations based on the given context.
Summarize or explain the most relevant information from the provided context.

Context:
{context}

Question:
{question}

If the question is vague or unrelated to the context, respond with: "❌ No relevant answer found."

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Handle user prompt
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        db = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(
            "❌ Could not load vector store. Make sure to run `embed_data.py` first."
        )
        return

    docs = db.similarity_search(user_question, k=5)

    # Check if top document is meaningful
    top_doc = docs[0].page_content.strip() if docs else ""
    if not top_doc or len(top_doc) < 50:
        st.warning("❌ No relevant answer found.")
        return

    chain = get_conversational_chain()
    context = "\n\n".join([doc.page_content for doc in docs])

    response = chain.invoke({"input_documents": docs, "question": user_question})
    ai_response = (
        response.get("output_text")
        or response.get("text")
        or "⚠️ Unexpected response format."
    )
    st.write("**Answer:**", ai_response)


# Streamlit UI
def main():
    st.set_page_config("Capillary Chatbot", page_icon="🤖")
    st.header("📘 Capillary Knowledge Base Chatbot 🤖")

    user_question = st.text_input("Ask a question based on the documentation...")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("ℹ️ Info")
        st.markdown(
            """
- Uses pre-processed `.json` files stored in `data/`
- Embeddings must be created separately using `embed_data.py`
- Queries answered using Gemini Pro via LangChain
        """
        )
        st.success("Vector index loaded from `faiss_index/`")


if __name__ == "__main__":
    main()
