import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import requests

# Funciones existentes
def load_pdf_to_faiss(pdf_path, vector_store=None):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if vector_store is None:
        vector_store = FAISS.from_documents(texts, embeddings)
    else:
        vector_store.add_documents(texts)
    return vector_store

def query_ollama(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 100):
    url = "http://localhost:11434/v1/completions"
    data = {"model": model, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        choices = result.get("choices", [])
        text_value = choices[0].get("text") if choices else None
        return text_value
    except requests.RequestException as e:
        return f"Error al interactuar con Ollama: {e}"

# Interfaz Streamlit
st.title("Sistema RAG con PDF y Preguntas")
st.write("Carga un archivo PDF, ingresa una pregunta y obtén respuestas generadas por el modelo.")

# Carga de PDF
uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")
vector_store = None

if uploaded_file is not None:
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    vector_store = load_pdf_to_faiss("uploaded_file.pdf")
    st.success("PDF cargado y procesado exitosamente.")

# Input para la pregunta
question = st.text_input("Escribe tu pregunta:")
if st.button("Generar Respuesta") and vector_store and question:
    docs = vector_store.similarity_search(question, k=7)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Contexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
    respuesta = query_ollama(model="llama3", prompt=prompt)
    st.text_area("Respuesta Generada:", respuesta, height=200)