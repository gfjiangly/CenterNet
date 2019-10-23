#!/bin/bash

cd src
# train
python main.py ctdet --exp_id dota_rg_resdcn18 --dataset dota --arch resdcn_18 --batch_size 48  --lr 5e-4 --num_workers 0 --resume --val_intervals 200
# test
python test.py ctdet --exp_id dota_rg_resdcn18 --dataset dota --arch resdcn_18 --keep_res --resume
# flip test
python test.py ctdet --exp_id dota_rg_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test
# multi scale test
python test.py ctdet --exp_id dota_rg_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
