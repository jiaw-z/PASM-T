#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch  --nproc_per_node 7 --master_port 1234 functions/train.py \
--cfg cfgs/sceneflow/pasm.yaml \
--data_root '/data1/StereoMatching/SceneFlow/driving' \
--data_name sceneflow \
--trainlist ./filenames/sceneflow_train.txt \
--testlist ./filenames/sceneflow_test.txt \
#--debug True \
#--apex True \
#--load_path /home/user/zhangjw/CONSTRUCTING/ST-TRANS/logs/sttr/sceneflow/04-07-11:34/ckpt_epoch_3.tar \
#--output_dir /home/user/zhangjw/CONSTRUCTING/ST-TRANS/results/secneflow \
#--trainlist ./filenames/sceneflow_train.txt \
#--testlist ./filenames/sceneflow_test.txt \

#--load_path /data2/zhangjw/CONSTRUCTING/ST-PASM-TRANS-4.20-15.43/logs/pasm/sceneflow/04-20-19:23/ckpt_epoch_5.tar \
#--is_continue True \

#--load_path /home/user/zhangjw/CONSTRUCTING/St-transformer/logs/sttr/sceneflow/03-30-20:41/ckpt_epoch_14.tar \
#--is_continue True \

#--trainlist ./filenames/sceneflow_train.txt \
#--testlist ./filenames/sceneflow_test.txt \

