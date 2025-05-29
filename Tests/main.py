from fastapi import FastAPI, Depends, HTTPException
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# snapshot_download(
#     repo_id="sentence-transformers/all-MiniLM-L6-v2",
#     local_dir="E:\Model",  # you can set any path
#     local_dir_use_symlinks=False  # makes actual copies instead of symlinks
# )

# print("Embedding shape:", model.encode("Test").shape)

app  = FastAPI()

# Allow requests from your frontend
origins = [
    "http://localhost:3000",  # your Next.js frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all origins (less secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = SentenceTransformer("E:/Model")
class ProductData(BaseModel):
    whatToEmbed: str
 
@app.post("/embed")
async def add_embeddings(data:ProductData):
        searchData = data.whatToEmbed
        embeddings = model.encode(searchData)
        return {"success":True,"status":200,"embeddings": embeddings.tolist()}

@app.get('/search')
async def search(query: str):
        embeddings = model.encode(query)
        return {"embeddings": embeddings}

        
