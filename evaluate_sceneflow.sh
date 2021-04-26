#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --master_port 88888  --nproc_per_node 1 functions/evaluate.py \
--cfg cfgs/sceneflow/pasm.yaml \
--data_root '/data1/StereoMatching/SceneFlow/driving' \
--data_name sceneflow \
--load_path /data2/zhangjw/CONSTRUCTING/ST-PASM-TRANS/logs/pasm/sceneflow/04-20-09:58/ckpt_epoch_12.tar \
--output_dir ./results/sceneflow \
--trainlist ./filenames/sceneflow_train.txt \
--testlist ./filenames/sceneflow_test_select.txt \
#--debug True \

#--trainlist ./filenames/sceneflow_train.txt \
#--testlist ./filenames/sceneflow_test.txt \
#/data2/zhangjw/CONSTRUCTING/ST-PASM-4.15-20.27-qk.scale/filenames/sceneflow-valid-select.txt