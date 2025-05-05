from transformers import BlipProcessor, BlipForConditionalGeneration

from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import torch.nn.functional as F
from fuzzywuzzy import fuzz
import numpy as np

class BLIP2Model:
    def __init__(self, classnames, 
                 blip_model_name="Salesforce/blip-image-captioning-base",
                 batch_size=10,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.classnames = classnames
        self.batch_size = batch_size
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from fuzzywuzzy import fuzz
        from PIL import Image
        self.processor = BlipProcessor.from_pretrained(blip_model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(device)
        self.model.eval()
        self.fuzz = fuzz
        self.Image = Image
        self.device = device

    def _batched(self, lst, max_prompt_tokens=512):
        """Yield batches of classnames, размер которых не превышает max_prompt_tokens"""
        i = 0
        while i < len(lst):
            # попробуем набрать батч как можно больше, но не превышая max_prompt_tokens
            for batch_size in range(self.batch_size, 0, -1):
                class_batch = lst[i:i+batch_size]
                class_list = ", ".join(class_batch)
                prompt = f"Classes: {class_list}. What is shown in the image?"
                input_ids = self.processor.tokenizer(prompt, return_tensors="pt")["input_ids"]
                if input_ids.shape[1] <= max_prompt_tokens:
                    yield class_batch
                    i += batch_size
                    break
            else:
                # Если даже один класс не помещается — ошибка
                raise ValueError(f"Class '{lst[i]}' слишком длинный для prompt!")

    def predict(self, image_paths):
        preds = []
        for img_path in image_paths:
            img = self.Image.open(img_path).convert("RGB")
            best_score = -float('inf')
            best_idx = -1
            for class_batch in self._batched(self.classnames, self.batch_size):
                class_list = ", ".join(class_batch)
                prompt = f"This image shows one of the following: {class_list}. What is shown in the image?"
                input_ids = self.processor.tokenizer(prompt, return_tensors="pt")["input_ids"]
                if input_ids.shape[1] > 512:
                    continue
                inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(**inputs, max_new_tokens=15)
                    caption = self.processor.decode(out[0], skip_special_tokens=True).lower()
                batch_scores = [self.fuzz.token_set_ratio(caption, c) for c in class_batch]
                max_score = max(batch_scores)
                max_idx_in_batch = batch_scores.index(max_score)
                if max_score > best_score:
                    best_score = max_score
                    best_idx = self.classnames.index(class_batch[max_idx_in_batch])
                preds.append(self.classnames[best_idx])
        return preds

    def predict_proba(self, image_paths):
        all_probs = []
        for img_path in image_paths:
            img = self.Image.open(img_path).convert("RGB")
            # Сохраняем все скоринги по всем классам
            scores = [None] * len(self.classnames)
            for batch_i, class_batch in enumerate(self._batched(self.classnames, self.batch_size)):
                class_list = ", ".join(class_batch)
                prompt = f"This image shows one of the following: {class_list}. What is shown in the image?"
                input_ids = self.processor.tokenizer(prompt, return_tensors="pt")["input_ids"]
                if input_ids.shape[1] > 512:
                    continue
                inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(**inputs, max_new_tokens=15)
                    caption = self.processor.decode(out[0], skip_special_tokens=True).lower()
                batch_scores = [self.fuzz.token_set_ratio(caption, c) for c in class_batch]
                # Заполняем правильные позиции в общем списке скорингов
                for i, score in enumerate(batch_scores):
                    class_idx = batch_i * self.batch_size + i
                    scores[class_idx] = score
            # Если для какого-то класса не было скоринга (например, prompt пропущен) — ставим минимальное значение
            for i, v in enumerate(scores):
                if v is None:
                    scores[i] = 0
            # В нормализованный массив вероятностей через softmax
            scores_np = np.array(scores) / 100.0  # fuzzywuzzy в диапазоне 0–100
            exp_scores = np.exp(scores_np)
            probs = exp_scores / np.sum(exp_scores)
            all_probs.append(probs)
        return all_probs
    

    def get_name(self):
        return "BLIP2"