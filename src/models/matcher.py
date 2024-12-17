import os
import io
import torch
import base64
from PIL import Image
from torchvision import transforms

from langchain_chroma import Chroma
from src.models.resnet import Resnet, FeatureExtractor





class GalleryDB(Chroma):
    def __init__(self, collection_name, persist_directory=""):
        super().__init__(collection_name=collection_name,
                         persist_directory=persist_directory,
                         collection_metadata={"hnsw:space": "cosine"})


    def add_embedding(self, embedding, id, document):
        self._collection.upsert(embeddings=[embedding], ids=[id], documents=[document])



