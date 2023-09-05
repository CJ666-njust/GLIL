#! /bin/bash
# python train.py
CUDA_VISIBLE_DEVICES=0,1,2,3, python -m torch.distributed.launch --master_port 33333 --nproc_per_node=5 --use_env train_mult_gpu.py --tag='inat' --data_path=dataset/inat2017  --lr=5e-7 --batch_size=24 --epochs=100 --image_size=304 --log_name=train_log_inat.txt
