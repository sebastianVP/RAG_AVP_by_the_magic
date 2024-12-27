import requests
from sklearn.metrics import precision_score, recall_score
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def load_pdf_to_faiss(pdf_path, vector_store=None):
    """
    Carga un PDF y lo indexa usando FAISS.
    """
    #print("--------------Load PDF to faiss----------")

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


# Evaluación de recuperación
def evaluate_retrieval(true_relevant_docs, retrieved_docs):
    relevant_retrieved = set(true_relevant_docs).intersection(set(retrieved_docs))
    precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
    recall = len(relevant_retrieved) / len(true_relevant_docs) if true_relevant_docs else 0
    return {
        "Precision@k": precision,
        "Recall": recall
    }

# Evaluación de generación
def evaluate_generation(reference_answers, generated_answers):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(generated_answers, reference_answers, avg=True)
    bleu_scores = [
        sentence_bleu([ref.split()], gen.split())
        for ref, gen in zip(reference_answers, generated_answers)
    ]
    return {
        "Rouge-1": rouge_scores['rouge-1']['f'],
        "Rouge-2": rouge_scores['rouge-2']['f'],
        "Rouge-L": rouge_scores['rouge-l']['f'],
        "BLEU": np.mean(bleu_scores)
    }

# Función principal para evaluar RAG
def evaluate_rag(pdf_path, question, true_answer, true_relevant_docs):
    # Crear el índice FAISS
    vector_store = load_pdf_to_faiss(pdf_path)
    
    # Recuperar documentos relevantes
    retrieved_docs = vector_store.similarity_search(question, k=7)
    retrieved_content = [doc.page_content for doc in retrieved_docs]

    # Métricas de recuperación
    retrieval_metrics = evaluate_retrieval(true_relevant_docs, retrieved_content)

    # Generar respuesta
    context = "\n".join(retrieved_content)
    prompt = f"Contexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
    generated_answer = query_ollama(model="llama3", prompt=prompt)
    print("--------GENERATED_ANSWER--------")
    print(generated_answer)
    # Métricas de generación
    generation_metrics = evaluate_generation([true_answer], [generated_answer])

    # Resultados
    print("Métricas de Recuperación:", retrieval_metrics)
    print("Métricas de Generación:", generation_metrics)
    print("Respuesta Generada:", generated_answer)

# Ejecución
if __name__ == "__main__":
    pdf_path = "hoja_de_vida.pdf"  # Archivo PDF con datos
    question = "¿Quién es la presidenta del Perú en 2024?"
    true_answer = ("Según la información proporcionada, Dina Ercilia Boluarte Zegarra "
                   "es la presidenta del Perú desde el 7 de diciembre de 2022, por lo que al "
                   "momento de responder a esta pregunta (2024), sigue siendo la Presidenta del Perú.")
    true_relevant_docs = ["Dina Ercilia Boluarte Zegarra es la presidenta del Perú desde el 7 de diciembre de 2022."]

    evaluate_rag(pdf_path, question, true_answer, true_relevant_docs)