from metrics.f1_metric import F1Metric
from metrics.top_k_accuracy import TopKAccuracyMetric
from terminaltables import AsciiTable
import torch

from models.declip_model import DeCLIPModel
from models.coca_model import CoCaModel
from models.flava_model import FlavaZSModel
from models.open_clip_model import OpenCLIP


debug = False

import pandas as pd
from datasets.base_dataset import BaseDataset
from metrics.f1_metric import F1Metric
from metrics.top_k_accuracy import TopKAccuracyMetric
from models.open_clip_model import OpenCLIP

class CSVDataset(BaseDataset):
    def __init__(self, csv_path, class_filter=None, split='test', domain_filter=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['split'] == split]
        if domain_filter:
            self.data = self.data[self.data['domain'] == domain_filter]
        if class_filter:
            self.data = self.data[self.data['class_name'].isin(class_filter)]

    def get_samples(self, n_samples=None):
        if n_samples:
            sampled_data = self.data.sample(n_samples)
        else:
            sampled_data = self.data

        image_paths = sampled_data['image_path'].tolist()
        labels = sampled_data['class_name'].tolist()
        return image_paths, labels

    def get_classnames(self):
        return sorted(self.data['class_name'].unique().tolist())



def run_benchmark(model_class, dataset, ds_name, n_samples=None, batch_size=1):
    classnames = dataset.get_classnames()
    model = model_class(classnames)
    image_paths, labels = dataset.get_samples(n_samples)

    f1_metric = F1Metric(average='weighted')
    top1 = TopKAccuracyMetric(k=1)
    preds = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_preds = model.predict(batch_paths)
        preds.extend(batch_preds)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    f1_score = f1_metric.compute(preds, labels)
    top1_score = top1.compute([[p] for p in preds], labels)

    return [
        model_class.__name__,
        ds_name,
        f"{f1_score:.4f}",
        f"{top1_score:.4f}"
    ]


def main():
    csv_path = 'merged_dataset.csv'
    domains = ['CUB_200_2011_split', 'dtd_split', 'fungi_clef_2022_split']
    model_classes = [
        OpenCLIP,
        DeCLIPModel,
        CoCaModel,
        FlavaZSModel,
    ]

    for domain in domains:
        print(f"\n=== Domain: {domain} ===")
        table_data = [["Model", "Dataset", "F1", "Top1 Acc"]]
        dataset = CSVDataset(csv_path=csv_path, split='test', domain_filter=domain)
        for model_class in model_classes:
            result_row = run_benchmark(model_class, dataset, ds_name=domain, n_samples=None, batch_size=1)
            table_data.append(result_row)
        table = AsciiTable(table_data)
        print(table.table)
 


if __name__ == "__main__":
    main()