# AECRNet-MindSpore

This is the MindSpore version of Contrastive Learning for Compact Single Image Dehazing, CVPR2021

The official PyTorch implementation, pretrained models and examples are available at [https://github.com/GlassyWu/AECR-Net](https://github.com/GlassyWu/AECR-Net)

## Abstract

Single image dehazing is a challenging ill-posed problem due to the severe information degeneration. However, existing deep learning based dehazing methods only adopt clear images as positive samples to guide the training of dehazing network while negative information is unexploited. Moreover, most of them focus on strengthening the dehazing network with an increase of depth and width, leading to a significant requirement of computation and memory. In this paper, we propose a novel contrastive regularization (CR) built upon contrastive learning to exploit both the information of hazy images and clear images as negative and positive samples, respectively. CR ensures that the restored image is pulled to closer to the clear image and pushed to far away from the hazy image in the representation space. Furthermore, considering trade-off between performance and memory storage, we develop a compact dehazing network based on autoencoder-like (AE) framework. It involves an adaptive mixup operation and a dynamic feature enhancement module, which can benefit from preserving information flow adaptively and expanding the receptive field to improve the network’s transformation capability, respectively. We term our dehazing network with autoencoder and contrastive regularization as AECR-Net. The extensive experiments on synthetic and real-world datasets demonstrate that our AECR-Net surpass the state-of-the-art approaches.

![image-20210413200215378](https://github.com/Booooooooooo/AECRNet-MindSpore/blob/main/images/model.png)

## Results

![image-20210413200307113](https://github.com/Booooooooooo/AECRNet-MindSpore/blob/main/images/trade-off.png)

![image-20210413200327940](https://github.com/Booooooooooo/AECRNet-MindSpore/blob/main/images/results.png)

![image-visual](https://github.com/Booooooooooo/AECRNet-MindSpore/blob/main/images/visual.png)

## Dependencies

- Python == 3.7.5
- MindSpore: https://www.mindspore.cn/install