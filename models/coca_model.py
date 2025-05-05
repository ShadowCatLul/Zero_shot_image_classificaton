from models.base_model import BaseModel
import torch
import open_clip

from PIL import Image

class CoCaModel(BaseModel):
    def __init__(self, classnames, model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()
        self.classnames = classnames
        self.prompts = [f"a photo of a {name.lower()}" for name in classnames]
        with torch.no_grad():
            text_tokens = self.tokenizer(self.prompts).to(self.device)
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def predict(self, image_paths, batch_size=16):
        preds = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            images = torch.stack(images).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                pred_indices = logits.argmax(dim=-1).cpu().tolist()
                batch_preds = [self.classnames[i] for i in pred_indices]
                preds.extend(batch_preds)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return preds

    def predict_proba(self, image_paths, topk=5, batch_size=16):
        topk_preds = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            images = torch.stack(images).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                probs = logits.cpu().tolist()
                for p in probs:
                    topk_indices = sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:topk]
                    topk_labels = [self.classnames[i] for i in topk_indices]
                    topk_preds.append(topk_labels)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return topk_preds

    def get_name(self):
        return "CoCa-OpenCLIP"
    