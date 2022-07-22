#!/bin/bash -ex
python train.py --outdir=/results/ --cfg=rmtv_google_shoes --data=/root/small_dataset \
	--gpus=8 --batch=32 --gamma=5 --metrics=None --data_image_resolution=512 --use_alpha_background=True;
