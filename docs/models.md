

Google Drive links for pre-trained checkpoints can be found [here.](https://drive.google.com/drive/folders/1y8QPJofiCokLmk8e-F6fZA2qrPmP8xO4?usp=sharing)

Brief descriptions of models and the commands used to train them are found below.

---

# FFHQ

**ffhq512-64.pkl**

FFHQ 512, trained with neural rendering resolution of 64x64.

```.bash
# Train with FFHQ from scratch with raw neural rendering resolution=64, using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/datasets/FFHQ_512.zip \
  --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True
```

**ffhq512-128.pkl**

Fine-tune FFHQ 512, with neural rendering resolution of 128x128.

```.bash
# Second stage finetuning of FFHQ to 128 neural rendering resolution.
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/datasets/FFHQ_512.zip \
  --resume=ffhq-64.pkl \
  --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True --neural_rendering_resolution_final=128 --kimg=2000
```

## FFHQ Rebalanced

Same as the models above, but fine-tuned using a rebalanced version of FFHQ that has a more uniform pose distribution. Compared to models trained on standard FFHQ, these models should produce better 3D shapes and better renderings from steep angles.

**ffhqrebalanced512-64.pkl**

```.bash
# Finetune with rebalanced FFHQ at rendering resolution 64.
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/datasets/FFHQ_rebalanced_512.zip \
  --resume=ffhq-64.pkl \
  --gpus=8 --batch=32 --gamma=1
```

**ffhqrebalanced512-128.pkl**
```.bash
# Finetune with rebalanced FFHQ at 128 neural rendering resolution.
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/datasets/FFHQ_rebalanced_512.zip \
  --resume=ffhq-rebalanced-64.pkl \
  --gpus=8 --batch=32 --gamma=1 --neural_rendering_resolution_final=128
```

# AFHQ Cats

**afhqcats512-128.pkl**

```.bash
# Train with AFHQ, finetuning from FFHQ with ADA, using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=afhq --data=~/datasets/afhq.zip \
  --resume=ffhq-64.pkl \
  --gpus=8 --batch=32 --gamma=5 --aug=ada --neural_rendering_resolution_final=128 --gen_pose_cond=True --gpc_reg_prob=0.8
```


# Shapenet

**shapenetcars128-64.pkl**

```.bash
# Train with Shapenet from scratch, using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=shapenet --data=~/datasets/cars_train.zip \
  --gpus=8 --batch=32 --gamma=0.3
```

---

# Training Tips

## Output image resolution
Output image resolution is determined by the size of images in the dataset. If you are using ```dataset_tool.py```, you can specify image resolution with 

```python dataset_tool.py --source images --dest dataset.zip --resolution 512x512```

## Neural rendering resolution
Neural rendering resolution is the resolution at which we volumetrically render, and it is independent of your output image size. In general, low neural rendering resolutions (e.g. 64) are faster at training and at inference. Higher neural rendering resolutions (e.g. 128) are more compute intensive but have less aliasing and produce more detailed shapes. For most models, we train at neural rendering resolution of 64 and optionally continue training with a neural rendering resolution of 128.

To train with a static neural rendering resolution of 64:
```.bash
python train.py \
  --neural_rendering_resolution_initial=64 \
  ...
```

To train with a neural rendering resolution that changes gradually increases from 64 to 128 over 1 million images:
```.bash
python train.py \
  --neural_rendering_resolution_initial=64 \
  --neural_rendering_resolution_final=128 \
  --neural_rendering_resolution_fade_kimg=1000 \
  ...
```

Please see **Two-stage training** (Section 3 of the supplemental) for additional details.

## Gamma
```--gamma``` is an important hyperparameter for ensuring stability of GAN training. The best value of ```gamma``` may vary widely between datasets. If you have nothing to go on, ```--gamma=5``` is often a safe choice. If your batch size is small, or if your images are large, you may need more regularization (higher gamma); if your output image size is small (e.g. Shapenet 128x128), you might be able to get away with smaller values of gamma. If your training is mode-collapsed (all images are very similar) you may benefit from increasing gamma; if your training is stable and you are looking for better image quality, try decreasing gamma.