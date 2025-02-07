import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from langchain.embeddings.openai import OpenAIEmbeddings

class EmbeddingUtils:
    def __init__(self, openai_api_key):
        self.text_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def create_text_embedding(self, text: str):
        return self.text_model.embed_query(text)

    def create_image_embedding(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        # Flatten to a 1D numpy array
        return image_features[0].cpu().numpy()
