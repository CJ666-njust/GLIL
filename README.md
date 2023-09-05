# Attention Enhances Close-up Perception: Global-Local Iterative Learning for Fine-Grained Visual Categorization

### Framework

![1693882664020](image/README/1693882664020.png)

### Requirements

torch==1.10.2

torchvision==0.11.3

numpy

tqdm

tensorboard

### Usage

#### 1. Download pre-trained ViT models

Google ViT-B-16: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth

#### 2.datasets

CUB-200-2011: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

Stanford Dogs: [Stanford Dogs dataset for Fine-Grained Visual Categorization](http://vision.stanford.edu/aditya86/ImageNetDogs/)

NABirds: [CCUB NABirds 700 Dataset Competition (allaboutbirds.org)](https://dl.allaboutbirds.org/nabirds)

iNat2017: [inat_comp/2017 at master · visipedia/inat_comp (github.com)](https://github.com/visipedia/inat_comp/tree/master/2017)

#### 3.Train

To train GLIL, you can run by "python train.py" on single gpu or "bash train.sh" on multiple gpus.
