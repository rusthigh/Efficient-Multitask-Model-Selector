
# Efficient Multimodal Multitask Model Selector
This is the PyTorch implementation of the paper **Efficient Multimodal Multitask Model Selector**.

## Overview
We propose an efficient multi-task model selector (EMMS), which transforms diverse label formats, such as categories, texts, and bounding boxes of different downstream tasks into a unified noisy label embedding. Extensive experiments on 5 downstream tasks with 24 datasets show that EMMS is fast, effective, and generic.

![Alt text](EMMS.png)

## Getting Started
Follow the guide below to get started.

### Data Preparation
- Download downstream datasets to ./data/*.

### Pipeline of Model selection using transferability
Extract features of target data using pretrained models and different labels of target data. Image classification tasks and image caption tasks have different pipelines.

- Image classification with CNN and ViT models:
  - `python forward_feature_CNN.py`
  - `python forward_feature_ViT.py`

- Image caption:
  - `python forward_feature_caption.py`

### Compute transferability scores
Compute transferability scores using EMMS and assess the effectiveness using model feature and F-labels:
- Image classification:
  - `python evaluate_metric_cls_cpu_CNN.py`  
  - `python evaluate_metric_cls_cpu_ViT.py`

- Image caption:
  - `python evaluate_metric_caption_cpu.py`

For other baselines such as LogME, use the metric parameter to replace.

## Contact
For any questions, email the new owner at rusthigh@gmail.com