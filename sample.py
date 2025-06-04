from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
import requests

# snapshot_download(
#     repo_id="openai/clip-vit-base-patch32",
#     local_dir=r"E:\ImageModel",  
#     local_dir_use_symlinks=False  
# )

model = CLIPModel.from_pretrained(r"E:\ImageModel")
processor = CLIPProcessor.from_pretrained(r"E:\ImageModel")
print("✅ Model and processor loaded successfully.")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open("008_LandCruiser250FE.jpg")

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    image_embeddings = model.get_image_features(**inputs)

image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)

print("Image embeddings shape:", image_embeddings.shape)
print("Image embeddings :", image_embeddings)