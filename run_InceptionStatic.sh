#!/bin/bash
# RUN 1
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[0]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t0_320x180jpg1fps_bs100_rmsprop_noaug_run1' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0 
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[2,3,0,5]' --test-subjects '[1]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t1_320x180jpg1fps_bs100_rmsprop_noaug_run1' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[3,0,1,5]' --test-subjects '[2]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t2_320x180jpg1fps_bs100_rmsprop_noaug_run1' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[0,1,2,5]' --test-subjects '[3]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t3_320x180jpg1fps_bs100_rmsprop_noaug_run1' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[0,1,2,3]' --test-subjects '[5]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t5_320x180jpg1fps_bs100_rmsprop_noaug_run1' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
# # RUN 2
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[0]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t0_320x180jpg1fps_bs100_rmsprop_noaug_run2' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0 
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[2,3,0,5]' --test-subjects '[1]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t1_320x180jpg1fps_bs100_rmsprop_noaug_run2' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[3,0,1,5]' --test-subjects '[2]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t2_320x180jpg1fps_bs100_rmsprop_noaug_run2' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[0,1,2,5]' --test-subjects '[3]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t3_320x180jpg1fps_bs100_rmsprop_noaug_run2' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[0,1,2,3]' --test-subjects '[5]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t5_320x180jpg1fps_bs100_rmsprop_noaug_run2' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
# # RUN 3
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[0]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t0_320x180jpg1fps_bs100_rmsprop_noaug_run3' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0 
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[2,3,0,5]' --test-subjects '[1]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t1_320x180jpg1fps_bs100_rmsprop_noaug_run3' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[3,0,1,5]' --test-subjects '[2]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t2_320x180jpg1fps_bs100_rmsprop_noaug_run3' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
# python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[0,1,2,5]' --test-subjects '[3]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t3_320x180jpg1fps_bs100_rmsprop_noaug_run3' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[0,1,2,3]' --test-subjects '[5]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t5_320x180jpg1fps_bs100_rmsprop_noaug_run3' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
# RUN 4
python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[0]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t0_320x180jpg1fps_bs100_rmsprop_noaug_run4' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0 
python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[2,3,0,5]' --test-subjects '[1]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t1_320x180jpg1fps_bs100_rmsprop_noaug_run4' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[3,0,1,5]' --test-subjects '[2]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t2_320x180jpg1fps_bs100_rmsprop_noaug_run4' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
 python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.01 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[0,1,2,5]' --test-subjects '[3]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t3_320x180jpg1fps_bs100_rmsprop_noaug_run4' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[0,1,2,3]' --test-subjects '[5]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t5_320x180jpg1fps_bs100_rmsprop_noaug_run4' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
# RUN 5
python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[0]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t0_320x180jpg1fps_bs100_rmsprop_noaug_run5' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0 
python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[2,3,0,5]' --test-subjects '[1]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t1_320x180jpg1fps_bs100_rmsprop_noaug_run5' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[3,0,1,5]' --test-subjects '[2]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t2_320x180jpg1fps_bs100_rmsprop_noaug_run5' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[0,1,2,5]' --test-subjects '[3]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t3_320x180jpg1fps_bs100_rmsprop_noaug_run5' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  
python main.py --data-path 'data/jpg_320_180_1fps/' --model 'inception_4d_input' --input-width 320 --input-height 180 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'rmsprop' --lr 0.001 --batch-size 100 --seq-length 1 --seq-stride 1 --round-to-batch 1 --train-subjects '[0,1,2,3]' --test-subjects '[5]' --val-subjects '[4]' --subjects-overview 'horse_subjects.csv' --nb-workers 1 --image-identifier 'trainfromlayer249_dense512_v4_t5_320x180jpg1fps_bs100_rmsprop_noaug_run5' --test-run 0 --nb-input-dims 4 --val-fraction 0 --data-type 'rgb' --aug-flip 0 --aug-crop 0 --aug-light 0  

