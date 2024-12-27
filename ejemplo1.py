import requests

def query_ollama(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 100):
    """
    Interactúa con Ollama para realizar consultas a un modelo.

    Args:
        model (str): Nombre del modelo en Ollama (por ejemplo, "llama2").
        prompt (str): Texto de entrada o pregunta para el modelo.
        temperature (float): Nivel de creatividad del modelo (0-1).
        max_tokens (int): Máximo número de tokens a generar.

    Returns:
        str: Respuesta generada por el modelo.
    """
    url = "http://localhost:11434/v1/completions"  # Endpoint de Ollama

    # Configuración de la solicitud
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        # Realizar la solicitud a Ollama
        response = requests.post(url, json=data)
        response.raise_for_status()  # Lanzar un error si la solicitud falla

        # Decodificar y retornar la respuesta
        result = response.json()
        # Acceder al texto dentro de "choices" usando `get`
        choices = result.get("choices", [])  # Obtiene la lista o un valor predeterminado (vacío)
        text_value = choices[0].get("text") if choices else None  # Accede al "text" del primer elemento si existe

        # Mostrar el resultado
        
        if text_value:
            #print("Texto obtenido:", text_value)
            return text_value
        else:
            #print("No se encontró el texto.")
            return None
        #return text_value
    except requests.RequestException as e:
        return f"Error al interactuar con Ollama: {e}"

# Ejemplo de uso
if __name__ == "__main__":
    modelo = "llama3"
    pregunta = "¿Qué es la inteligencia artificial?"
    pregunta = "¿Quién es el ultimo presidente de Perú?"


    respuesta = query_ollama(model=modelo, prompt=pregunta)
    print(f"Pregunta: {pregunta}")
    print(f"Respuesta: {respuesta}")
