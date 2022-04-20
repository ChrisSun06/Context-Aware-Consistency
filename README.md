# Semi-supervised semantic segmantation
Our implementation is based on the following two papers:

*Xin Lai<sup>\*</sup>, Zhuotao Tian<sup>\*</sup>, Li Jiang, Shu Liu, Hengshuang Zhao, Liwei Wang, Jiaya Jia* [**Semi-supervised Semantic Segmentation with Directional Context-aware Consistency**](https://jiaya.me/papers/semiseg_cvpr21.pdf)  (CVPR 2021). [[Paper]](https://jiaya.me/papers/semiseg_cvpr21.pdf)

[Xiaokang Chen](https://charlescxk.github.io/)1, [Yuhui Yuan](https://scholar.google.com/citations?user=PzyvzksAAAAJ&hl=zh-CN)2, [Gang Zeng](https://www.cis.pku.edu.cn/info/1177/1378.htm)1, [Jingdong Wang](https://jingdongwang2017.github.io/) **[Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision](https://arxiv.org/abs/2106.01226)** 

We use the source code from the first paper: https://github.com/dvlab-research/Context-Aware-Consistency to build our implementation for the second paper. For our implementation for the second paper, we built upon its source code https://github.com/charlesCXK/TorchSemiSeg but made moderate modifications for it to run on the code base for the first paper.

We also include below an instruction to run & train the model on *Cityscapes* dataset which is not mentioned in the source code of the first paper.

# Get Started
## Environment
The repository is tested on Ubuntu 18.04.3 LTS, Python 3.6.9, PyTorch 1.6.0 and CUDA 10.2
```
pip install -r requirements.txt
```

## Datasets Preparation

#### Pascal VOC

1. Firstly, download the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) Dataset, and the extra annotations from [SegmentationClassAug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0).

2. Make a directory called pretrained.

   ````
   mkdir pretrained

2. Extract the above compression files into your pretrained, and make it follow the directory tree as below.

```
-Context-Aware-Consistency
	-pretrained
        -VOCtrainval_11-May-2012
            -VOCdevkit
                -VOC2012
                    -Annotations
                    -ImageSets
                    -JPEGImages
                    -SegmentationClass
                    -SegmentationClassAug
                    -SegmentationObject
```

3. Set 'data_dir' in the config file into '[YOUR_PATH]/VOCtrainval_11-May-2012'.

#### Cityscapes

1. Download the Cityscapes dataset using below commands:

   ```
   wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<username>&password=<password>$&submit=Login' https://www.cityscapes-dataset.com/login/
   
   wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
   
   wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
   ```

   Note: To download the dataset will require you to sign-up for the website: https://www.cityscapes-dataset.com/ , after sign-up, simply input your username and password to the top command above.

2. Extract the two downloaded files, you should get two folders: `gtFine` and `leftImg8bit`, put the two folders under `pretrained` as well.

3. So far only 372 labels and 744 labels under `dataloaders/city_splits0` can be used, other files contains unmatched images which can be quickly fixed by replacing all `labelTrainIds` to `labelIds`

#### Training

You should download the PyTorch ResNet101 or ResNet50 ImageNet-pretrained weight, and put it into the 'pretrained/' directory using the following commands.

```
cd Context-Aware-Consistency
cd pretrained
wget https://download.pytorch.org/models/resnet50-19c8e357.pth # ResNet50
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth # ResNet101
```

Run the following commands for training.

- train the model on the 1/8 labeled data (the 0-th data list) of PASCAL VOC/Cityscape with CAC w/ ResNet50:
```
python3 train.py --config configs/voc_cac_deeplabv3+_resnet50_1over8_datalist0.json
python3 train.py --config configs/city_cac_deeplabv3+_resnet50_1over8_datalist0.json
```

- train the model on the 1/8 labeled data (the 0-th data list) of PASCAL VOC with CAC w/ ResNet101:
```
python3 train.py --config configs/voc_cac_deeplabv3+_resnet101_1over8_datalist0.json
```

- train the model on the 1/8 labeled data (the 0-th data list) of PASCAL VOC/Cityscapes with regular CPS:

```
python3 train_semiseg.py --config configs/voc_cac_deeplabv3+_resnet50_1over8_datalist0.json
python3 train_semiseg.py --config configs/city_cac_deeplabv3+_resnet50_1over8_datalist0.json
```

- train the model on the 1/8 labeled data (the 0-th data list) of PASCAL VOC/Cityscapes with CutMix CPS:

```
python3 train_semiseg_cutmix.py --config configs/voc_cac_deeplabv3+_resnet50_1over8_datalist0.json
python3 train_semiseg_cutmix.py --config configs/city_cac_deeplabv3+_resnet50_1over8_datalist0.json
```

## Pre-trained Models For CAC

For your convenience, you can download some of the pre-trained models of CAC from [Here](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155154502_link_cuhk_edu_hk/EpHdT2JFT11FpsUc4jHE3CoB2wUZ5tQo_W0QzzqHdNtF-A?e=yx2Xha).

# Related Repositories

This repository is forked from https://github.com/dvlab-research/Context-Aware-Consistency, we thank the authors for their implementation.

The code's original author borrowed codes from below repositories:

- **CCT** https://github.com/yassouali/CCT.

- **MoCo** at https://github.com/facebookresearch/moco. 
- **Deeplabv3+** at https://github.com/jfzhang95/pytorch-deeplab-xception.
- **Semseg** at https://github.com/hszhao/semseg
