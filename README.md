# [TIFS'25] Semantic Contextualization of Face Forgery: A New Definition, Dataset, and Detection Method (SO-DFD)

This repository contains the official PyTorch implementation of the paper **"[Semantic Contextualization of Face Forgery: A New Definition, Dataset, and Detection Method](https://ieeexplore.ieee.org/document/10948473)"** by Mian Zou, Baosheng Yu, Yibing Zhan, Siwei Lyu, and Kede Ma.

☀️ If you find this work useful for your research, please kindly star our repo and cite our paper! ☀️

### TODO
We are working hard on the following items.

- [x] Release [arXiv paper](https://arxiv.org/abs/2405.08487)
- [x] Release training codes
- [x] Release inference codes
- [ ] Release checkpoints 
- [ ] Release datasets

## 📁 Datasets
Follow the links below to download the datasets (🛡️ Copyright of the datasets belongs to their original providers, and you may be asked to fill out some forms before downloading):

|  [FF++](https://github.com/ondyari/FaceForensics) | [CDF(v2)](https://github.com/yuezunli/celeb-deepfakeforensics)|
|:-:|:-:|
[FSh](https://github.com/ondyari/FaceForensics/blob/master/dataset/FaceShifter/README.md) | [DF-1.0](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master) | 
| [FFSC]()  |[DFDC (test set of the full version, not the Preview)](https://ai.meta.com/datasets/dfdc/) |

**Note**: If a separate test set is explicitly provided in the dataset project page, please download the test set. Otherwise, if no specific split is mentioned, download the full dataset. 

### Preprocessing (see [instructions](https://github.com/MZMMSEC/SJEDD/tree/main/preprocessing))

1) Extract the frames, and then detect and crop the faces (Optional for video datasets)

2) Rearrange the data for the test experiments


## 🚀 Quick Start

### 1. Installation of base reqiurements
 - python == 3.8
 - PyTorch == 1.13
 - Miniconda
 - CUDA == 11.7

### 2. Download the pretrained model and our model
|      Model       |    Training Dataset   |                                                        Download                                                                | |
|:----------------:|:----------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|
| SO-Xception | FFSC  | [Google Drive]() |⬜|
| SO-ViT-B | FFSC  | [Google Drive]() |⬜|
| SO-ViT-B | FF++  | [Google Drive]() |⬜|

After downloading these checkpoints, put them into the folder ``pretrained``.

### 3. Training 

### 4. Inference on the test sets



## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@article{zou2025semantic,
  title={Semantic contextualization of face forgery: A new definition, dataset, and detection method},
  author={Zou, Mian and Yu, Baosheng and Zhan, Yibing and Lyu, Siwei and Ma, Kede},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  pages={4512-4524},
  vol={20}
}
@article{zou2025sjedd,
  title={Semantics-oriented multitask learning for DeepFake detection: A joint embedding approach},
  author={Zou, Mian and Yu, Baosheng and Zhan, Yibing and Lyu, Siwei and Ma, Kede},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
```
