from datasets.cub_dataset import CUBDataset
from datasets.dtd_dataset import DTDDataset
from datasets.fungi_clef_2022_dataset import FungiCLEF2022Dataset
from metrics.f1_metric import F1Metric
from metrics.top_k_accuracy import TopKAccuracyMetric
from terminaltables import AsciiTable
from models.base_model import BaseModel
import torch

from models.declip_model import DeCLIPModel
from models.coca_model import CoCaModel
##from models.lit_model import LiTModel
from models.flava_model import FlavaZSModel
from models.nano_vlm_model import NanoVLMModel
from models.open_clip_model import OpenCLIP
from models.blip2_model import BLIP2Model


debug = True

def run_benchmark(model_class, dataset, n_samples=10000):
    classnames = dataset.get_classnames()
    model = model_class(classnames)
    image_paths, labels = dataset.get_samples(n_samples)
    f1_metric = F1Metric(average='weighted')
    top1 = TopKAccuracyMetric(k=1)
    preds = model.predict(image_paths)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    probs = model.predict_proba(image_paths)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    f1_score = f1_metric.compute(preds, labels)
    top1_score = top1.compute([[p] for p in preds], labels)


    if debug:
       
        print("Метка из датасета:", labels[0])
        print("Класс из модели:", model.classnames[0])
        print("Все классы модели:", model.classnames)
        print("Предсказание:", preds[0])
        #print("Top-5 предсказания:", topk_preds[0])
    
    return [
        model_class.__name__,
        dataset.__class__.__name__,
        f"{f1_score:.4f}",
        f"{top1_score:.4f}"
    ]

def main():
    datasets = [CUBDataset(), DTDDataset(), FungiCLEF2022Dataset()]
    model_classes = [
        OpenCLIP,
       ## DeCLIPModel,
        #CoCaModel,
        # LiTModel,
      #  FlavaZSModel,
        #BLIP2Model,
        # NanoVLMModel
    ]
   
    table_data = [
        ["Model", "Dataset", "F1", "Top1 Acc"]
    ]

    for model_class in model_classes:
        for dataset in datasets:
            result_row = run_benchmark(model_class, dataset)
            table_data.append(result_row)

    table = AsciiTable(table_data)
    print(table.table)

if __name__ == "__main__":
    main()