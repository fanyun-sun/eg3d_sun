#!/bin/bash -ex
python train.py --outdir=/results --cfg=shapenet --data=/root/eg3d_sun/dataset_preprocessing/shapenet_chairs/chairs_128.zip --data_image_resolution 128  --gpus 8 --batch=32 --gamma=0.3 
