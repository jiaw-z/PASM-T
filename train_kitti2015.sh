#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --master_port 88888  --nproc_per_node 7 functions/train.py \
--cfg cfgs/kitti/pasm.yaml \
--data_root /data1/StereoMatching/KITTI-2015 \
--data_name kitti2015 \
--trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
--load_path /data2/zhangjw/CONSTRUCTING/ST-PASM-TRANS-4.20-15.43/logs/pasm/sceneflow/04-22-21:34/ckpt_epoch_11.tar \
#--debug True \
#--load_path /home/user/zhangjw/CONSTRUCTING/ST-TRANS/logs/sttr/kitti/debug/ckpt_epoch_60.tar \
#--is_continue True \
#--debug True \
