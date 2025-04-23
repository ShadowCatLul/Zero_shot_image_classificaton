
import torch
from PIL import Image
from torchvision import transforms
from models.base_model import BaseModel
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "Sense_GVT", "DeCLIP", "prototype")))

from .Sense_GVT.DeCLIP.prototype.model.declip import DECLIP
from .Sense_GVT.DeCLIP.prototype.model.image_encoder.visual_transformer import VisualTransformer
from .Sense_GVT.DeCLIP.prototype.model.text_encoder.text_transformer import TextTransformer
from .Sense_GVT.DeCLIP.prototype.model.utils.text_utils.simple_tokenizer import SimpleTokenizer

class DeCLIPModel(BaseModel):
    def __init__(self, classnames, model_ckpt_path="vitb32.pth.tar", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classnames = classnames

        # 1. Инициализация архитектуры
        self.vision_model = VisualTransformer(
            input_resolution=224,
            patch_size=32,
            width=768,
            layers=12,
            heads=12,
            embed_dim=3072,
            checkpoint=None
        )
        self.text_model = TextTransformer(
            embed_dim=3072,
            context_length=77,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
            positional_embedding_flag=True,
            checkpoint=None,
            bpe_path="models/Sense_GVT/DeCLIP/prototype/model/utils/text_utils/bpe_simple_vocab_16e6.txt.gz",
            text_encode_type='Transformer',
            text_model_utils={}
        )

        self.model = DECLIP(
            image_encode=self.vision_model,
            text_encode=self.text_model,
            use_allgather=False,
            feature_dim=3072
        ).to(self.device)



        # 2. Загрузка весов
        state_dict = torch.load(model_ckpt_path, map_location=self.device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        for param in self.model.visual.conv1.parameters():
            param.requires_grad = False

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

        # 4. Токенизация
        self.tokenizer = SimpleTokenizer()
        self.context_length = 77

        texts = [f"a photo of a {name}" for name in self.classnames]
        with torch.no_grad():
            self.text_features = self.model.encode_text(texts)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def predict(self, image_paths, batch_size=16):
        preds = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            if self.preprocess:
                images = [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            else:
                images = [Image.open(p).convert("RGB") for p in batch_paths]
            images = torch.stack(images).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(images)
                logits = (self.model.logit_scale.exp() * image_features @ self.text_features.t()).softmax(dim=-1)
                pred_indices = logits.argmax(dim=-1).cpu().tolist()
                batch_preds = [self.classnames[i] for i in pred_indices]
                preds.extend(batch_preds)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return preds

    def predict_proba(self, image_paths, topk=5, batch_size=16):
        topk_preds = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            if self.preprocess:
                images = [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            else:
                images = [Image.open(p).convert("RGB") for p in batch_paths]
            images = torch.stack(images).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(images)
                logits = (self.model.logit_scale.exp() * image_features @ self.text_features.t()).softmax(dim=-1)
                probs = logits.cpu().tolist()
                for p in probs:
                    topk_indices = sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:topk]
                    topk_labels = [self.classnames[i] for i in topk_indices]
                    topk_preds.append(topk_labels)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return topk_preds

    def get_name(self):
        return "DeCLIP"