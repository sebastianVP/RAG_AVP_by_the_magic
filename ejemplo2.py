import requests
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import warnings
warnings.filterwarnings('ignore')

def query_ollama(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 100):
    """
    Interactúa con Ollama para realizar consultas a un modelo.
    """
    url = "http://localhost:11434/v1/completions"  # Endpoint de Ollama
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        choices = result.get("choices", [])
        text_value = choices[0].get("text") if choices else None
        return text_value
    except requests.RequestException as e:
        return f"Error al interactuar con Ollama: {e}"

def load_pdf_to_faiss(pdf_path):
    """
    Carga un PDF y lo indexa usando FAISS.
    """
    # Cargar el PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Dividir el texto en fragmentos
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Crear embeddings y un índice FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)

    return vector_store

def query_rag(pdf_path, question):
    """
    Realiza una pregunta utilizando un sistema RAG.
    """
    # Indexar el contenido del PDF
    vector_store = load_pdf_to_faiss(pdf_path)

    # Buscar los fragmentos más relevantes
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Crear el prompt combinado
    prompt = f"Contexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
    
    # Consultar a Ollama
    respuesta = query_ollama(model="llama3", prompt=prompt)
    return respuesta

# Ejemplo de uso
if __name__ == "__main__":
    pdf_path = "hoja_de_vida.pdf"  # Ruta al PDF
    pregunta = "¿Qué es el ultimo presidente del Peru?"
    
    respuesta = query_rag(pdf_path, pregunta)
    print(f"Pregunta: {pregunta}")
    print(f"Respuesta: {respuesta}")
