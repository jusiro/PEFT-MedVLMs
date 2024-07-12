### Foundation models for medical imaging: Hands on!

This project includes required codebase for an **introductory hands-on sesion on vision-language foundation models for medical image analysis**. Also, we include the particular details for Datasets prepation in this file (***see below***).

## Overview

**Foundation models are large-scale networks, pre-trained on large, heterogeneus sources, which can be efficeintly adapted to a variety of downstream tasks**.

In particular, this tutorial is focused on **vision specialized medical foundation models**. This is, **modality-specialized** pre-trained networks. For example, we will work over models pre-trained uniquely on histology data. Such specialized vision models provide specially efficient transferability to new tasks/datasets. For example, they can be adapted using only **few labeled** examples (so-called **shots**), and requiring **minimal parameter tuning**.

In this tutorial, due to the large resources required for pre-trained, **we will focus on the adaptation stage**. Nevertheless, we will introduce toy examples to introduce the student to typical losses and pipelines employed. Regarding the adaptation, **you will learn how to archieve state-of-the-art performance on your image classification dataset employing minimum data and computing resources by leveraging recently introduced vision-language foundation models**. 

### Index

    1. Vision-language models
        1.1. Introduction, application, VLMs, and Transformers library
        1.2. Contrastive text-image pre-training
        1.3. Zero-shot classification
            1.3.1. Single prompt
            1.3.2. Prompt ensemble
        1.4. Few-shot black-box Adapters
            1.4.1. Linear Probe
            1.4.2. CLIP-Adapter
            1.4.3. Zero-shot Linear Probe
            1.4.4. Class-Adaptive Linear Probe (CLAP)
        1.5. Few-shot Parameter-Efficient Fine-Tuning
            1.5.1. Selective (Bias, Affine)
            1.5.2. Additive (LoRA)

## Datasets 

**Required time: 15-20 minutes**
​
#### Vision-Language Models 

In the following, we describe the required datasets for the activities carried out for vision-language medical foundation models. We will mostly focus on the adaptation of pre-trained models, for which you requiere to prepare the following data sources.
​
- [SICAPv2](https://data.mendeley.com/datasets/9xxm58dvs3/2) This datasets, originary from the paper [1] contains histology images of tumor prostate tissue labeled at the image level into different severity grades, so-called Gleason grades. The task considered is image level multi-class classification.

The download process is fast, you only need around **2 minutes**. Then, locally extract the folder `SICAPv2`, and upload it to the Saturn Cloud Project. The upload process to `./local_data/datasets/` might take around **10-15 minutes**, and you require around 2,0 Gb of disk memory. Finally, unzip by running `unzip SICAPv2.zip` in a command window, and ready!
​

## Folder Overview


Once you have download all datasets and models, your project should present the following structure:

```
    .
    ├── DLMI24_HO_FM/
    │   ├── local_data/
    │   │   └──datasets/
    │   │      └── SICAPv2
    │   └── vlms/
    │       └── ...
    ├── 1_1_VLMs_Introduction.ipynb
    ├── 1_2_VLMs_Pretraining.ipynb
    ├── 1_3_ZeroShot.ipynb
    ├── 1_4_BlackBox.ipynb
    ├── 1_5_PEFT.ipynb
    └── README.MD
```

## References


[1] Silva-Rodríguez, J., Colomer, A., Sales, M. A., Molina, R., & Naranjo, V. (2020). Going deeper through the Gleason scoring scale : An automatic end-to-end system for histology prostate grading and cribriform pattern detection. Computer Methods and Programs in Biomedicine.
