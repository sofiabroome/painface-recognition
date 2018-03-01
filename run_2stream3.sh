#!/bin/bash
# python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model '2stream_5d' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-layers 4 --nb-lstm-units 32 --nb-conv-filters 16 --nb-dense-units 64 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 8 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-horses '[1,2,3,5]' --test-horses '[0]' --val-horses ['4'] --nb-workers 1 --image-identifier 'add_v4_t0_4hl_128jpg2fps_seq10_bs8_MAG_adadelta_flipcropshade_run3' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model '2stream_5d' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-layers 4 --nb-lstm-units 32 --nb-conv-filters 16 --nb-dense-units 64 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 8 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-horses '[2,3,0,5]' --test-horses '[1]' --val-horses ['4'] --nb-workers 1 --image-identifier 'add_v4_t1_4hl_128jpg2fps_seq10_bs8_MAG_adadelta_flipcropshade_run3' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model '2stream_5d' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-layers 4 --nb-lstm-units 32 --nb-conv-filters 16 --nb-dense-units 64 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 8 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-horses '[3,0,1,5]' --test-horses '[2]' --val-horses ['4'] --nb-workers 1 --image-identifier 'add_v4_t2_4hl_128jpg2fps_seq10_bs8_MAG_adadelta_flipcropshade_run3' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model '2stream_5d' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-layers 4 --nb-lstm-units 32 --nb-conv-filters 16 --nb-dense-units 64 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 8 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-horses '[0,1,2,5]' --test-horses '[3]' --val-horses ['4'] --nb-workers 1 --image-identifier 'add_v4_t3_4hl_128jpg2fps_seq10_bs8_MAG_adadelta_flipcropshade_run3' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1 
python main.py --data-path 'data/jpg_128_128_2fps/' --of-path 'data/jpg_128_128_16fps_OF_magnitude_cv2/' --model '2stream_5d' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-layers 4 --nb-lstm-units 32 --nb-conv-filters 16 --nb-dense-units 64 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --batch-size 8 --seq-length 10 --seq-stride 10 --round-to-batch 1 --train-horses '[0,1,2,3]' --test-horses '[5]' --val-horses ['4'] --nb-workers 1 --image-identifier 'add_v4_t5_4hl_128jpg2fps_seq10_bs8_MAG_adadelta_flipcropshade_run3' --test-run 0 --nb-input-dims 5 --val-fraction 0 --aug-flip 1 --aug-crop 1 --aug-light 1 
