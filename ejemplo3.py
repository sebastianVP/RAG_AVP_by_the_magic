import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def filter_paragraphs(paragraphs, min_length=30, max_length=500):
    """
    Filtra los párrafos por tamaño.

    Args:
        paragraphs (list): Lista de párrafos extraídos.
        min_length (int): Longitud mínima del párrafo.
        max_length (int): Longitud máxima del párrafo.

    Returns:
        list: Lista de párrafos filtrados.
    """
    return [
        para for para in paragraphs
        if min_length <= len(para.get_text()) <= max_length
    ]

def split_large_content(content, chunk_size=500):
    """
    Divide el contenido en fragmentos de tamaño limitado.

    Args:
        content (str): Texto completo.
        chunk_size (int): Máximo número de caracteres por fragmento.

    Returns:
        list: Lista de fragmentos.
    """
    return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

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

def load_pdf_to_faiss(pdf_path, vector_store=None):
    """
    Carga un PDF y lo indexa usando FAISS.
    """
    print("--------------Load PDF to faiss----------")

    # Cargar el PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    #Sprint("Documents",documents)
    # Dividir el texto en fragmentos
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Crear embeddings y un índice FAISS (o agregar al existente)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if vector_store is None:
        vector_store = FAISS.from_documents(texts, embeddings)
    else:
        vector_store.add_documents(texts)

    return vector_store

def load_web_to_faiss(url, vector_store=None, max_paragraphs=20):
    """
    Extrae el texto principal de una página web, filtra el contenido y lo indexa en FAISS.
    """
    print("--------------Load WEB to faiss----------")

    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        filtered_paragraphs = filter_paragraphs(paragraphs)

        # Limitar el número de párrafos procesados
        limited_paragraphs = filtered_paragraphs[:max_paragraphs]
        content = "\n".join([para.get_text() for para in limited_paragraphs])

        # Dividir en fragmentos
        fragments = split_large_content(content)

        # Crear documentos
        documents = [Document(page_content=frag, metadata={"source": url}) for frag in fragments]

        # Crear embeddings y un índice FAISS
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if vector_store is None:
            vector_store = FAISS.from_documents(documents, embeddings)
        else:
            vector_store.add_documents(documents)

        return vector_store
    except requests.RequestException as e:
        print(f"Error al extraer contenido de la página web: {e}")
        return vector_store
    

def query_rag_with_web_and_pdf(pdf_path, url, question):
    """
    Realiza una pregunta utilizando un sistema RAG, combinando contenido de PDFs y páginas web.
    """
    # Crear un índice FAISS
    vector_store = None

    # Agregar contenido del PDF al índice
    vector_store = load_pdf_to_faiss(pdf_path, vector_store)

    # Agregar contenido de la web al índice
    vector_store = load_web_to_faiss(url, vector_store)

    # Buscar los fragmentos más relevantes en el índice
    docs = vector_store.similarity_search(question, k=7)
    context = "\n".join([doc.page_content for doc in docs])
    # Imprimir el contexto 
    print("Contexto completo:\n")
    for i, doc in enumerate(docs, start=1):
        print(f"Documento {i}:")
        print("-" * 40)  # Separador
        print(doc.page_content)
        print("-" * 40)
        print("\n")  # Espacio entre documentos
        # Crear el prompt combinado
    prompt = f"Contexto:\n{context}\n\nPregunta: {question}\nRespuesta:"

    # Consultar a Ollama
    respuesta = query_ollama(model="llama3", prompt=prompt)
    return respuesta

# Ejemplo de uso
if __name__ == "__main__":
    pdf_path = "hoja_de_vida.pdf"  # Ruta al PDF
    url = "https://www.gob.pe/presidencia"  # URL de la página web
    pregunta = "¿Qué es el presidente del Peru en el 2024?"

    respuesta = query_rag_with_web_and_pdf(pdf_path, url, pregunta)
    print(f"Pregunta: {pregunta}")
    print(f"Respuesta: {respuesta}")