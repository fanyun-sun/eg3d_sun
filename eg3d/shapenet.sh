#!/bin/bash -ex
python train.py --outdir=/root/training-runs --cfg=shapenet --data=/root/cars_128 --data_image_resolution 128  --gpus 8 --batch=32 --gamma=0.3 
