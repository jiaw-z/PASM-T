#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --master_port 88888  --nproc_per_node 1 functions/evaluate.py \
--cfg cfgs/kitti/pasm.yaml \
--data_root /data1/StereoMatching/KITTI-2015 \
--data_name kitti2015 \
--load_path /data2/zhangjw/CONSTRUCTING/ST-PASM-TRANS-4.20-15.43/logs/pasm/kitti/04-26-09:27/ckpt_epoch_60.tar \
--output_dir ./results/kitti \
--trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
--debug True \

#--trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
#/data2/zhangjw/CONSTRUCTING/ST-PASM-4.13-16.02/logs/pasm/kitti/04-14-14:54/ckpt_epoch_90.tar
#/data2/zhangjw/CONSTRUCTING/ST-PASM-4.16-21.04/logs/pasm/kitti/04-17-16:32/ckpt_epoch_25.tar
#/data2/zhangjw/CONSTRUCTING/ST-PASM-TRANS/logs/pasm/kitti/04-19-23:00/ckpt_epoch_130.tar