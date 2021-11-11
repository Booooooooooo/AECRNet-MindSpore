# AECRNet-MindSpore

This is the MindSpore version of [Contrastive Learning for Compact Single Image Dehazing, CVPR2021](https://arxiv.org/abs/2104.09367)

The official PyTorch implementation, pretrained models and examples are available at [https://github.com/GlassyWu/AECR-Net](https://github.com/GlassyWu/AECR-Net)

The DCN_v2 module is based on https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/centernet. We thank the authors for sharing the codes. 
## Abstract

Single image dehazing is a challenging ill-posed problem due to the severe information degeneration. However, existing deep learning based dehazing methods only adopt clear images as positive samples to guide the training of dehazing network while negative information is unexploited. Moreover, most of them focus on strengthening the dehazing network with an increase of depth and width, leading to a significant requirement of computation and memory. In this paper, we propose a novel contrastive regularization (CR) built upon contrastive learning to exploit both the information of hazy images and clear images as negative and positive samples, respectively. CR ensures that the restored image is pulled to closer to the clear image and pushed to far away from the hazy image in the representation space. Furthermore, considering trade-off between performance and memory storage, we develop a compact dehazing network based on autoencoder-like (AE) framework. It involves an adaptive mixup operation and a dynamic feature enhancement module, which can benefit from preserving information flow adaptively and expanding the receptive field to improve the network’s transformation capability, respectively. We term our dehazing network with autoencoder and contrastive regularization as AECR-Net. The extensive experiments on synthetic and real-world datasets demonstrate that our AECR-Net surpass the state-of-the-art approaches.

![image-20210413200215378](https://github.com/Booooooooooo/AECRNet-MindSpore/blob/main/images/model.png)

## Dependencies

- Python == 3.7.5
- MindSpore: https://www.mindspore.cn/install
- numpy
- ModelArts：https://console.huaweicloud.com/modelarts/?region=cn-north-4#/dashboard

## Train

### Prepare data

We use [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-standard) training set as our training data.

### Train on ModelArts

[ModelArts](https://support.huaweicloud.com/modelarts/index.html) is a one-stop AI development platform that enables developers and data scientists of any skill level to rapidly build, train, and deploy models anywhere, from the cloud to the edge.  Feel free to sign up and get hands on!

1.  Create OBS bucket and prepare dataset.
2.  VGG pre-trained on ImageNet is used in our contrastive loss. Due to copyright reasons, the pre-trained VGG cannot be shared publicly. 
3.  We use PyCharm toolkit to help with the training process. You could find tutorial [here](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0021.html). Or you could start training following this [tutorial](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0080.html).

### Train on GPU

`python train_wCR.py --device_target GPU --dir_data LOCATION_OF_DATA --test_every 1 --filename aecrnet_id1 --lr 0.0002 --epochs 300 --patch_size 240 --data_train RESIDE --neg_num 4 --contra_lambda 20`

## Evaluation
`python eval.py --dir_data LOCATION_OF_DATA/Dense_Haze --batch_size 1 --test_only --ext "img" --data_test Dense --ckpt_path ckpt/aecrnet_id1.ckpt`

## Results

![image-20210413200307113](https://github.com/Booooooooooo/AECRNet-MindSpore/blob/main/images/trade-off.png)

![image-20210413200327940](https://github.com/Booooooooooo/AECRNet-MindSpore/blob/main/images/results.png)

![image-visual](https://github.com/Booooooooooo/AECRNet-MindSpore/blob/main/images/visual.png)

## Citation
Please kindly cite the references in your publications if it helps your research:
```@inproceedings{wu2021contrastive,
      title={Contrastive Learning for Compact Single Image Dehazing}, 
      author={Haiyan Wu and Yanyun Qu and Shaohui Lin and Jian Zhou and Ruizhi Qiao and Zhizhong Zhang and Yuan Xie and Lizhuang Ma},
      year={2021},
      booktitle={Computer Vision and Pattern Recognition},
}
