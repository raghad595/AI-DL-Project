import streamlit as st
import spacy
from spacy import displacy
from spacy.cli import download

def main():
    st.title("Named Entity Recognition")
    st.write("Please enter some text")

    text = st.text_area("Input text")

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        with st.spinner("Downloading language model..."):
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

    if text:
        doc = nlp(text)
        html = displacy.render(doc, style="ent", page=True)
        st.write(html, unsafe_allow_html=True)
    else:
        st.write("Please enter text first")
