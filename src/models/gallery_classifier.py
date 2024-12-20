import torch
from PIL import Image
from torchvision import transforms
from torch.nn.functional import normalize

from langchain_chroma import Chroma
from src.models.resnet import FeatureExtractor




class GalleryDB(Chroma):
    def __init__(self, collection_name, persist_directory=""):
        super().__init__(collection_name=collection_name,
                         persist_directory=persist_directory,
                         collection_metadata={"hnsw:space": "cosine"})


    def add_embedding(self, embedding, id, document):
        # embedding = normalize(torch.from_numpy(embedding), dim=0).numpy()
        self._collection.upsert(embeddings=[embedding], ids=[id], documents=[document])



class VectorStore():
    def __init__(self, model_path, collection_name, out_path="", device="cpu"):
        self.device = device
        self.model = FeatureExtractor(model_path=model_path, num_classes=819).to(device)
        self.vector_store = GalleryDB(collection_name=collection_name, persist_directory=out_path)


    def generate_embedding(self, imgs):
        with torch.no_grad():
            embeddings = self.model(imgs.to(self.device))
        final_embedding = torch.mean(embeddings, dim=0)
        return final_embedding.cpu().numpy()


    def __fix_input(self, imgs, label=None, multiple_imgs=False):
        if label is not None:
            if isinstance(label, torch.Tensor):
                label = label.item()
            if not isinstance(label, str):
                label = str(label)

        if multiple_imgs:
            if isinstance(imgs, Image.Image):
                imgs = transforms.ToTensor()(imgs)
            if not isinstance(imgs, torch.Tensor):
                if isinstance(imgs, list) and  isinstance(imgs[0], Image.Image):
                    imgs = [transforms.ToTensor()(img) for img in imgs]
                if isinstance(imgs, list) and isinstance(imgs[0], torch.Tensor):
                    imgs = torch.stack(imgs)
            while len(imgs.shape) > 4 and imgs.shape[0] == 1:
                imgs = imgs.squeeze(0)
        return imgs if label is None else imgs, label


    def query(self, img):
        embedding = self.generate_embedding(img)
        toRet = self.vector_store.similarity_search_by_vector_with_relevance_scores(embedding, 1)
        id = int(toRet[0][0].page_content)
        similarity = 1 - toRet[0][1]
        return id, similarity


    def add_user(self, imgs, label):
        imgs, label = self.__fix_input(imgs=imgs, label=label, multiple_imgs=True)
        embedding = self.generate_embedding(imgs)
        self.vector_store.add_embedding(embedding, label, label)



class Matcher():
    def __init__(self, model_path, collection_name, threshold=None, out_path="", device="cpu"):
        self.threshold = threshold
        self.vector_store = VectorStore(model_path, collection_name, out_path, device)


    def match_train(self, img):
        id, similarity = self.vector_store.query(img)
        return id, similarity


    def match(self, img):
        id, similarity = self.vector_store.query(img)
        if similarity > self.threshold:
            return id, similarity
        else:
            return None, similarity


    def get_threshold(self):
        return self.threshold


    def add_user(self, imgs: list, label):
        self.vector_store.add_user(imgs, label)


    def set_threshold(self, threshold):
        self.threshold = threshold
