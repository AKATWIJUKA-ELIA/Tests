from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

snapshot_download(
    repo_id="openai/clip-vit-base-patch32",
    local_dir=r"E:\ImageModel",  # you can set any path
    local_dir_use_symlinks=False  # makes actual copies instead of symlinks
)

model = CLIPModel.from_pretrained("E:\ImageModel")
processor = CLIPProcessor.from_pretrained("E:\ImageModel")
print("✅ Model and processor loaded successfully.")
# model = SentenceTransformer("E:\ImageModel")
# print("Embedding shape:", model.encode("Test").shape)

# sentences = ["This is an example sentence", "Each sentence is converted"]

# embeddings = model.encode(sentences)
# print(embeddings)
