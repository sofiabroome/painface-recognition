#!/bin/bash
# RUN 1
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,3]' --val-subjects '[4]' --test-subjects '[0]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t0_seq10ss10_4hl_512ubs16_all_aug_run1' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,0,2,3]' --val-subjects '[4]' --test-subjects '[1]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t1_seq10ss10_4hl_512ubs16_all_aug_run1' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,0,1,3]' --val-subjects '[4]' --test-subjects '[2]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t2_seq10ss10_4hl_512ubs16_all_aug_run1' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,0]' --val-subjects '[4]' --test-subjects '[3]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t3_seq10ss10_4hl_512ubs16_all_aug_run1' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,3]' --val-subjects '[0]' --test-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val0_t4_seq10ss10_4hl_512ubs16_all_aug_run1' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[0,1,2,3]' --val-subjects '[4]' --test-subjects '[5]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t5_seq10ss10_4hl_512ubs16_all_aug_run1' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
# RUN 2
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,3]' --val-subjects '[4]' --test-subjects '[0]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t0_seq10ss10_4hl_512ubs16_all_aug_run2' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,0,2,3]' --val-subjects '[4]' --test-subjects '[1]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t1_seq10ss10_4hl_512ubs16_all_aug_run2' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,0,1,3]' --val-subjects '[4]' --test-subjects '[2]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t2_seq10ss10_4hl_512ubs16_all_aug_run2' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,0]' --val-subjects '[4]' --test-subjects '[3]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t3_seq10ss10_4hl_512ubs16_all_aug_run2' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,3]' --val-subjects '[0]' --test-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val0_t4_seq10ss10_4hl_512ubs16_all_aug_run2' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[0,1,2,3]' --val-subjects '[4]' --test-subjects '[5]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t5_seq10ss10_4hl_512ubs16_all_aug_run2' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
# RUN 3
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,3]' --val-subjects '[4]' --test-subjects '[0]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t0_seq10ss10_4hl_512ubs16_all_aug_run3' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,0,2,3]' --val-subjects '[4]' --test-subjects '[1]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t1_seq10ss10_4hl_512ubs16_all_aug_run3' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,0,1,3]' --val-subjects '[4]' --test-subjects '[2]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t2_seq10ss10_4hl_512ubs16_all_aug_run3' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,0]' --val-subjects '[4]' --test-subjects '[3]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t3_seq10ss10_4hl_512ubs16_all_aug_run3' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,3]' --val-subjects '[0]' --test-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val0_t4_seq10ss10_4hl_512ubs16_all_aug_run3' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[0,1,2,3]' --val-subjects '[4]' --test-subjects '[5]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t5_seq10ss10_4hl_512ubs16_all_aug_run3' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
# RUN 4
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,3]' --val-subjects '[4]' --test-subjects '[0]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t0_seq10ss10_4hl_512ubs16_all_aug_run4' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,0,2,3]' --val-subjects '[4]' --test-subjects '[1]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t1_seq10ss10_4hl_512ubs16_all_aug_run4' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,0,1,3]' --val-subjects '[4]' --test-subjects '[2]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t2_seq10ss10_4hl_512ubs16_all_aug_run4' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,0]' --val-subjects '[4]' --test-subjects '[3]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t3_seq10ss10_4hl_512ubs16_all_aug_run4' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,3]' --val-subjects '[0]' --test-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val0_t4_seq10ss10_4hl_512ubs16_all_aug_run4' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[0,1,2,3]' --val-subjects '[4]' --test-subjects '[5]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t5_seq10ss10_4hl_512ubs16_all_aug_run4' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
# RUN 5
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,3]' --val-subjects '[4]' --test-subjects '[0]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t0_seq10ss10_4hl_512ubs16_all_aug_run5' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,0,2,3]' --val-subjects '[4]' --test-subjects '[1]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t1_seq10ss10_4hl_512ubs16_all_aug_run5' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,0,1,3]' --val-subjects '[4]' --test-subjects '[2]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t2_seq10ss10_4hl_512ubs16_all_aug_run5' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,0]' --val-subjects '[4]' --test-subjects '[3]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t3_seq10ss10_4hl_512ubs16_all_aug_run5' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[5,1,2,3]' --val-subjects '[0]' --test-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val0_t4_seq10ss10_4hl_512ubs16_all_aug_run5' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --model 'rodriguez' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 512 --nb-dense-units 4096 --kernel-size 5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-subjects '[0,1,2,3]' --val-subjects '[4]' --test-subjects '[5]' --subjects-overview 'metadata/horse_subjects.csv' --image-identifier 'jpg128_2fps_val4_t5_seq10ss10_4hl_512ubs16_all_aug_run5' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 
