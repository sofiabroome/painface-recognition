#!/bin/bash
# RUN 1
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[0]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t0_run1' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[2,3,0,5]' --test-subjects '[1]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t1_run1' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[3,0,1,5]' --test-subjects '[2]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t2_run1' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[0,1,2,5]' --test-subjects '[3]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t3_run1' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[4]' --val-subjects '[0]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v0_t4_run1' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[0,1,2,3]' --test-subjects '[5]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t5_run1' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
# RUN 2
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[0]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t0_run2' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[2,3,0,5]' --test-subjects '[1]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t1_run2' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[3,0,1,5]' --test-subjects '[2]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t2_run2' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[0,1,2,5]' --test-subjects '[3]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t3_run2' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[4]' --val-subjects '[0]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v0_t4_run2' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[0,1,2,3]' --test-subjects '[5]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t5_run2' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
# RUN 3
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[0]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t0_run3' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[2,3,0,5]' --test-subjects '[1]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t1_run3' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[3,0,1,5]' --test-subjects '[2]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t2_run3' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[0,1,2,5]' --test-subjects '[3]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t3_run3' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[4]' --val-subjects '[0]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v0_t4_run3' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[0,1,2,3]' --test-subjects '[5]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t5_run3' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
# RUN 4
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[0]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t0_run4' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[2,3,0,5]' --test-subjects '[1]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t1_run4' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[3,0,1,5]' --test-subjects '[2]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t2_run4' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[0,1,2,5]' --test-subjects '[3]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t3_run4' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[4]' --val-subjects '[0]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v0_t4_run4' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[0,1,2,3]' --test-subjects '[5]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t5_run4' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
# RUN 5
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[0]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t0_run5' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[2,3,0,5]' --test-subjects '[1]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t1_run5' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[3,0,1,5]' --test-subjects '[2]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t2_run5' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[0,1,2,5]' --test-subjects '[3]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t3_run5' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[1,2,3,5]' --test-subjects '[4]' --val-subjects '[0]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v0_t4_run5' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model 'rodriguez_2stream' --input-width 128 --input-height 128 --nb-dense-units 4096 --nb-lstm-units 512 --nb-lstm-layers 4 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-labels 2  --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 4 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-subjects '[0,1,2,3]' --test-subjects '[5]' --val-subjects '[4]' --subjects-overview 'metadata/horse_subjects.csv' --nb-workers 1 --image-identifier '4hl512u_128x128jpg2fps_bs4_adadelta_all_aug_v4_t5_run5' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1  
