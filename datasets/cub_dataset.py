from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from typing import List, Tuple
import os
import random
import shutil

from metrics.f1_metric import F1Metric
from metrics.top_k_accuracy import TopKAccuracyMetric
from evaluation.evaluator import ModelEvaluator
from models.base_model import BaseModel
from datasets.base_dataset import BaseDataset

class CUBDataset(BaseDataset):
    def __init__(self, root="CUB_200_2011/CUB_200_2011"):
        self.root = root
        self.images_dir = os.path.join(root, "images")
        self.classes_file = os.path.join(root, "classes.txt")
        self.images_file = os.path.join(root, "images.txt")
        self.labels_file = os.path.join(root, "image_class_labels.txt")
        self.classnames = self._load_classnames()

    def _load_classnames(self) -> List[str]:
        with open(self.classes_file, "r") as f:
            return [line.strip().split(" ", 1)[1] for line in f.readlines()]

    def get_samples(self, n_samples=None) -> Tuple[List[str], List[str]]:
        with open(self.images_file, "r") as f:
            id2path = {int(line.split()[0]): line.split()[1] for line in f}
        with open(self.labels_file, "r") as f:
            id2label = {int(line.split()[0]): int(line.split()[1]) - 1 for line in f}
        image_ids = sorted(id2path.keys())
        if n_samples:
            image_ids = image_ids[:n_samples]
        image_paths = [os.path.join(self.images_dir, id2path[i]) for i in image_ids]
        labels = [self.classnames[id2label[i]] for i in image_ids]
        return image_paths, labels

    def get_classnames(self) -> List[str]:
        return self.classnames