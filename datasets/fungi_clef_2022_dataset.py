import os
import csv
from typing import List, Tuple
from .base_dataset import BaseDataset

class FungiCLEF2022Dataset(BaseDataset):
    def __init__(self, root="fungi_clef_2022", csv_file="DF20-train_metadata.csv", images_subdir="DF20_300"):
        self.root = root
        self.csv_path = os.path.join(root, csv_file)
        self.images_dir = os.path.join(root, images_subdir)
        self.samples, self.classnames = self._load_samples_and_classnames()

    @staticmethod
    def _shorten_scientific_name(full_name: str) -> str:
        # Берём только первые два слова (род и вид)
        return " ".join(full_name.split()[:2])

    def _load_samples_and_classnames(self) -> Tuple[List[Tuple[str, str]], List[str]]:
        samples = []
        all_classnames_set = set()
        with open(self.csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_name = self._shorten_scientific_name(row["scientificName"])
                all_classnames_set.add(class_name)
                img_path = os.path.join(self.images_dir, row["image_path"])
                if os.path.isfile(img_path):
                    samples.append((img_path, class_name))
        # classnames содержит все уникальные классы из CSV, даже если для них нет изображений
        classnames = sorted(all_classnames_set)
        return samples, classnames

    def get_samples(self, n_samples=None) -> Tuple[List[str], List[str]]:
        if n_samples:
            samples = self.samples[:n_samples]
        else:
            samples = self.samples
        if samples:
            image_paths, labels = zip(*samples)
            return list(image_paths), list(labels)
        else:
            return [], []

    def get_classnames(self) -> List[str]:
        return self.classnames