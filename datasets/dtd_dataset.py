import os
from typing import List, Tuple
from .base_dataset import BaseDataset

class DTDDataset(BaseDataset):
    def __init__(self, root="dtd", split="test1"):
        self.root = root
        self.images_dir = os.path.join(root, "images")
        self.labels_file = os.path.join(root, "labels", f"{split}.txt")
        self.classnames = sorted(os.listdir(self.images_dir))

    def get_samples(self, n_samples=None) -> Tuple[List[str], List[str]]:
        image_paths, labels = [], []
        with open(self.labels_file, "r") as f:
            for line in f:
                rel_path = line.strip()
                class_name = rel_path.split("/")[0]
                image_paths.append(os.path.join(self.images_dir, rel_path))
                labels.append(class_name)
        if n_samples:
            image_paths, labels = image_paths[:n_samples], labels[:n_samples]
        return image_paths, labels

    def get_classnames(self) -> List[str]:
        return self.classnames