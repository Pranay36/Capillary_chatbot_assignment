import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def load_and_combine_json(folder_path):
    combined_text = ""
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    print(f" Found {len(files)} JSON files in '{folder_path}'")
    for filename in tqdm(files, desc=" Reading JSON files"):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                title = item.get("title", "")
                body = item.get("content", "")
                combined_text += f"{title}\n{body}\n\n"
    return combined_text


def create_embeddings(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = splitter.split_text(text)

    print(f"Split into {len(chunks)} text chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print(" Creating embeddings...")

    # Show chunk-level progress bar
    embedded_chunks = []
    start_time = time.time()

    for chunk in tqdm(chunks, desc="🔄 Embedding chunks"):
        embedded_chunks.append(chunk)  # just track progress visually

    # Now pass full list at once
    vectorstore = FAISS.from_texts(embedded_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

    elapsed = time.time() - start_time
    print(f"Saved FAISS index with {len(chunks)} chunks.")
    print(f" Embedding completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    data_folder = "data"
    combined_text = load_and_combine_json(data_folder)
    create_embeddings(combined_text)
