import torch
from transformers import FlavaModel, FlavaProcessor
from PIL import Image
from models.base_model import BaseModel
from typing import List, Any

class FlavaZSModel(BaseModel):
    def __init__(self, classnames: List[str], device: str = None, prompt_template: str = "A photo of a {}."):
        self.classnames = classnames
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_template = prompt_template
        self.model = FlavaModel.from_pretrained("facebook/flava-full").to(self.device)
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        self.text_features = self._encode_texts(classnames)
    
    def _encode_texts(self, classnames: List[str]):
        texts = [self.prompt_template.format(c) for c in classnames]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model.text_model(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device)
            )
            text_embeds = outputs.pooler_output
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def _encode_images(self, images: List[Image.Image]):
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model.image_model(
                pixel_values=inputs["pixel_values"].to(self.device)
            )
            image_embeds = outputs.pooler_output
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds

    def predict(self, inputs: List[Any]) -> List[Any]:
        images = [Image.open(x).convert("RGB") if not isinstance(x, Image.Image) else x.convert("RGB") for x in inputs]
        image_embeds = self._encode_images(images)
        logits = image_embeds @ self.text_features.T
        preds = logits.argmax(dim=1).cpu().numpy()
        return [self.classnames[i] for i in preds]

    def predict_proba(self, inputs: List[Any]) -> List[List[float]]:
        images = [Image.open(x).convert("RGB") if not isinstance(x, Image.Image) else x.convert("RGB") for x in inputs]
        image_embeds = self._encode_images(images)
        logits = image_embeds @ self.text_features.T
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs.tolist()

    def get_name(self) -> str:
        return "FLAVA"