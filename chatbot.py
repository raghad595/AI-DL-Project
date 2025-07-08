import os
import streamlit as st
import faiss
import tempfile
import numpy as np
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer, util
import time

def main():
    api = "AIzaSyAJPLmGnk44PnmZ9D3PRg26lcL8AtCvL9Q"  # Note not the best practice to do! u should use dot env

    if api:
        genai.configure(api_key=api)
    else:
        st.error("API is not FOUND")

    model = genai.GenerativeModel("models/gemini-1.5-flash")  # it require internet connection
    def generate_text(text):
        response = model.generate_content(text)
        return response.text


    st.title("RAG with Vector DataBase and Gemini API")
    # Session state

    if "message" not in st.session_state:
        st.session_state.message = []

    for message in st.session_state.message:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    upload_file = st.file_uploader("Choose File", type=["pdf"])

    if upload_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(upload_file.read())
            tempfile_path = temp_file.name

        loader = PyPDFLoader(tempfile_path)
        documents = loader.load()

        model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

        text = [doc.page_content for doc in documents]
        embeddings = model.encode(text, show_progress_bar=True)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        st.success("PDF processed into the index")
        user_input = st.chat_input("Please enter ur Question")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            st.session_state.message.append({"role": "user", "content": user_input})

            question_embedding = model.encode([user_input])

            k = 1
            distance, indecies = index.search(question_embedding, k)

            similar_doc = [documents[i] for i in indecies[0]]

            context = ""

            for i, doc in enumerate(similar_doc):
                context += doc.page_content + "\n"

            prompt = f"you are a NLP expert who can define Lexical and Lemma from text and highlight it! with any visual based{context} and map it to question {user_input}"
            # prompt = f"{context}{user_input}"

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Generating Answer"):
                    try:
                        response_text = generate_text(prompt)
                    except Exception as e:
                        st.error(f"Rate limit hit. Please wait and try again. Error: {e}")
                        time.sleep(10)  # wait before retry
                        response_text = "Sorry, I couldn't process your request at the moment. Please try again later."
                    message_placeholder.markdown(f"{response_text}")
            st.session_state.message.append({"role": "assistant", "content": response_text})
    else:
        st.info("Please Upload ur file to chat with!!")
    st.markdown("Made with ❤️ by [Raghad]")
