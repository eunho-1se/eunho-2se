from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb, tiktoken, requests, json
from chromadb import Documents, EmbeddingFunction, Embeddings

app = FastAPI()

embedding_model = None
chroma_client = None

class UploadRequest(BaseModel):
    full_text: str
    chunk_size: int

class QueryRequest(BaseModel):
    query: str

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return embedding_model.encode(input).tolist()

@app.on_event("startup")
async def startup_event():
    global embedding_model, chroma_client

    embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    chroma_client = chromadb.PersistentClient()

def split_text(full_text, chunk_size):
    encoder = tiktoken.encoding_for_model('gpt-5')
    total_encoding = encoder.encode(full_text)
    total_token_count = len(total_encoding)
    text_list = []
    for i in range(0, total_token_count, chunk_size):
        chunk = total_encoding[i:i+chunk_size]
        decoded = encoder.decode(chunk)
        text_list.append(decoded)

    return text_list

@app.post("/upload")
def upload(request: UploadRequest):
    global embedding_model, chroma_client

    chunk_list = split_text(request.full_text, request.chunk_size)

    embeddings =embedding_model.encode(chunk_list)

    collection_name = 'language_collection'

    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass

    language_collection = chroma_client.create_collection(name=collection_name, embeddings_function=MyEmbeddingFunction())

    id_list = []
    for index in range(len(chunk_list)):
        id_list.append(f'{index}')

    language_collection.add(documents=chunk_list, ids=id_list)

    return {"ok": True, "chunks": len(chunk_list)}

@app.post("/answer")
def llm_response(request: QueryRequest):
    global embedding_model, chroma_client

    collection_name = "language_collection"

    language_collection = chroma_client.get_collection(name=collection_name, embedding_function=MyEmbeddingFunction())

    retrieved_documents = language_collection.query(query_texts=request.query, n_results=3)
    refer = retrieved_documents['documetns'][0]

    url = "http://ollama:11434/api/generate"

    payload = {
        "model": "gemma3:1b",
        "prompt": f'''You are a business analysis expert in Korea.
                        Please find answers to users' questions in our *Context*. If not, please direct them to the company.
                        Please organize your answers so users can understand them.
                        Please Answer after summerzie in Korean, don't use English.
                *Context*:
                {refer}
                *Question*: {request.query}

                Answer in Korean:''',
        "stream": False
    }

    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    return {"response": response.json()["response"]}
