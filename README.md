# Zero-shot классификации изображений

В этом репозитории находится реализация бенчмарка трансформерных архитектур для zero-shot классификации изображений, а также пайплайн обучения и все необходимые утилиты для тестирования и визулизации.


[ссылка на датасет](https://huggingface.co/datasets/ShadowCatLul/clef_fungi_dtd_for_zero-shot)

[ссылка на веса модели](https://huggingface.co/ShadowCatLul/OpenClip_for-zero-shot)

### test.py 
```
python test.py
```

Расчёт macro F1, Accuracy на тестовых выборках датасетов для каждой из архитектур:

| Домен           | Модель      | F1     | Top1 Accuracy |
|-----------------|------------|--------|---------------|
| CUB_200_2011    | OpenCLIP   | 0.7367 | 0.7502        |
|                 | DeCLIP     | 0.6627 | 0.6789        |
|                 | CoCa       | 0.6553 | 0.6819        |
|                 | FLAVA      | 0.0103 | 0.0253        |
|                 |            |        |               |
| DTD             | CoCa       | 0.7110 | 0.7148        |
|                 | OpenCLIP   | 0.6426 | 0.6556        |
|                 | DeCLIP     | 0.4635 | 0.5023        |
|                 | FLAVA      | 0.0307 | 0.0440        |
|                 |            |        |               |
| FungiCLEF_2022  | DeCLIP     | 0.3929 | 0.4166        |
|                 | OpenCLIP   | 0.1378 | 0.1542        |
|                 | CoCa       | 0.1315 | 0.1869        |
|                 | FLAVA      | 0.0558 | 0.0901        |


### data_analytics.ipynb

Экземпляры из датасетов и статистика:

| CUB  | DTD   | Fungi    |
| :---:   | :---: | :---: |
| ![image](https://github.com/ShadowCatLul/Zero_shot_image_classificaton/blob/master/sources/cub_example.png) | ![image](https://github.com/ShadowCatLul/Zero_shot_image_classificaton/blob/master/sources/dtd_example.png)   | ![image](https://github.com/ShadowCatLul/Zero_shot_image_classificaton/blob/master/sources/fungi_example.png)   |

### visualize.ipynb

Визуализация эмбедиингов изображений:

![image](https://github.com/ShadowCatLul/Zero_shot_image_classificaton/blob/master/sources/domains_graph.png)

![image](https://github.com/ShadowCatLul/Zero_shot_image_classificaton/blob/master/sources/t-sne_all_classes.png)


### model_tran_exp1.ipynb; model_tran_exp1.ipynb; prompting.ipynb - Эксперименты по улучшению качества классификации
1) model_tran_exp1.ipynb - fine-tune
2) model_tran_exp2.ipynb - fine-tune + bias-calibration
3) prompting.ipynb - ensemble-pompting + tta

Сводная таблица по результатам всех экспериментов:

| Домен          | Чекпоинт       | Эксперимент   | F1     | Top1 acc |
|----------------|---------------|--------------|--------|----------|
| CUB_200_2011   | Best_unseen   | 1            | 0.7220 | 0.7428   |
|                | Best_gzsl     | 1            | 0.7123 | 0.7315   |
|                | Best_unseen   | 2            | 0.7246 | 0.7440   |
|                | Best_gzsl     | 2            | 0.7083 | 0.7286   |
|                | Prompt        | 3            | 0.7696 | 0.7770   |
|                | Prompt + tta  | 3            | 0.7797 | 0.7880   |
|                | baseline      |              | 0.7367 | 0.7502   |
| dtd            | Best_unseen   | 1            | 0.5982 | 0.6194   |
|                | Best_gzsl     | 1            | 0.6024 | 0.6134   |
|                | Best_unseen   | 2            | 0.6042 | 0.6213   |
|                | Best_gzsl     | 2            | 0.5925 | 0.6042   |
|                | Prompt        | 3            | 0.6834 | 0.6931   |
|                | Prompt + tta  | 3            | 0.6924 | 0.7014   |
|                | baseline      |              | 0.6426 | 0.6556   |
| fungi_clef_2022| Best_unseen   | 1            | 0.1244 | 0.1786   |
|                | Best_gzsl     | 1            | 0.1334 | 0.1892   |
|                | Best_unseen   | 2            | 0.1245 | 0.1873   |
|                | Best_gzsl     | 2            | 0.1462 | 0.1955   |
|                | Prompt        | 3            | 0.1138 | 0.1892   |
|                | Prompt + tta  | 3            | 0.1162 | 0.1928   |
|                | baseline      |              | 0.1378 | 0.1542   |

