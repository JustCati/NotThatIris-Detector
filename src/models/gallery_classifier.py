import os
import torch
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



class VectorStore():
    def __init__(self, model_path, collection_name, out_path="", device="cpu"):
        self.device = device
        self.model = FeatureExtractor(Resnet.load_from_checkpoint(model_path, num_classes=819)).to(device)
        self.vector_store = GalleryDB(collection_name=collection_name, persist_directory=out_path)


    def generate_embedding(self, imgs):
        with torch.no_grad():
            embeddings = self.model(imgs.to(self.device))
        final_embedding = torch.mean(embeddings, dim=0)
        return final_embedding.cpu().numpy()


    def query(self, img):
        img = Image.open(img)
        embedding = self.generate_embedding(img)
        toRet = self.vector_store.similarity_search_by_vector_with_relevance_scores(embedding, 1)
        id = toRet[0][0].page_content
        similarity = 1 - toRet[0][1]
        return id, similarity


    def add_user(self, imgs, label):
        if isinstance(imgs, Image.Image):
            imgs = transforms.ToTensor()(imgs)
        if not isinstance(imgs, torch.Tensor):
            if isinstance(imgs[0], Image.Image) and isinstance(imgs, list):
                imgs = [transforms.ToTensor()(img) for img in imgs]
            if isinstance(imgs[0], torch.Tensor) and isinstance(imgs, list):
                imgs = torch.stack(imgs)

        if len(imgs.shape) == 5 and imgs.shape[0] == 1:
            imgs = imgs.squeeze(0)
        if isinstance(label, torch.Tensor):
            label = label.item()
        if not isinstance(label, str):
            label = str(label)

        embedding = self.generate_embedding(imgs)
        self.vector_store.add_embedding(embedding, label, label)



class Matcher():
    def __init__(self, model_path, collection_name, threshold, out_path="", device="cpu"):
        self.threshold = None if threshold == -1 else threshold
        self.vector_store = VectorStore(model_path, collection_name, out_path, device)


    def match(self, img):
        id, similarity = self.vector_store.query(img)
        if similarity > self.threshold:
            return id, similarity
        return None, None


    def add_user(self, imgs: list, label):
        self.vector_store.add_user(imgs, label)


    def change_threshold(self, threshold):
        self.threshold = threshold
