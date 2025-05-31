import gradio as gr
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import re
import os

# Load data and FAISS index
def load_data_and_index():
    docs_df = pd.read_pickle("docs_with_embeddings (1).pkl")  # Adjust path for HF Spaces
    embeddings = np.array(docs_df['embeddings'].tolist(), dtype=np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return docs_df, index

docs_df, index = load_data_and_index()

# Load SentenceTransformer
minilm = SentenceTransformer('all-MiniLM-L6-v2')

# Configure Gemini API using Hugging Face Secrets
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found. Please set it in Hugging Face Spaces secrets.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'[^\w\s.,;:>-]', ' ', text)
    text = ' '.join(text.split()).strip()
    return text

# Retrieve documents
def retrieve_docs(query, k=5):
    query_embedding = minilm.encode([query], show_progress_bar=False)[0].astype(np.float32)
    distances, indices = index.search(np.array([query_embedding]), k)
    retrieved_docs = docs_df.iloc[indices[0]][['label', 'text', 'source']]
    retrieved_docs['distance'] = distances[0]
    return retrieved_docs

# RAG pipeline integrated into respond function
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,  # Keeping top_p as an input, though Gemini doesn’t use it directly
):
    # Preprocess the user message
    preprocessed_query = preprocess_text(message)
    
    # Retrieve relevant documents
    retrieved_docs = retrieve_docs(preprocessed_query, k=5)
    context = "\n".join(retrieved_docs['text'].tolist())
    
    # Construct the prompt with system message, history, and RAG context
    prompt = f"{system_message}\n\n"
    for user_msg, assistant_msg in history:
        if user_msg:
            prompt += f"User: {user_msg}\n"
        if assistant_msg:
            prompt += f"Assistant: {assistant_msg}\n"
    prompt += (
        f"Query: {message}\n"
        f"Relevant Context: {context}\n"
        f"Generate a short, concise, and to-the-point response to the query based only on the provided context."
    )
    
    # Generate response with Gemini
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
    )
    answer = response.text.strip()
    if not answer.endswith('.'):
        last_period = answer.rfind('.')
        if last_period != -1:
            answer = answer[:last_period + 1]
        else:
            answer += "."
    
    # Yield the full response (no streaming, as Gemini API doesn’t support it here)
    yield answer

# Gradio Chat Interface
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(
            value="You are a medical AI assistant diagnosing patients based on their query, using relevant context from past records of other patients.",
            label="System message"
        ),
        gr.Slider(minimum=1, maximum=2048, value=150, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.75, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",  # Included but not used by Gemini
        ),
    ],
    title="🏥 Medical Chat Assistant",
    description="A chat-based medical assistant that diagnoses patient queries using AI and past records."
)

if __name__ == "__main__":
    demo.launch()