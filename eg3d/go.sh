#!/bin/bash -ex
python train.py --outdir=/results/ --cfg=rmtv_google_shoes --data=/root/small_dataset \
	--gpus=8 --batch=32 --gamma=5 --data_image_resolution=256 --use_alpha_background=True
