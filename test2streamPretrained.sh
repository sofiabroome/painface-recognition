#!/bin/bash
python main.py --data-path 'data/jpg_320_180_1fps/' --model '2stream_pretrained' --input-width 320 --input-height 180 --nb-labels 2 --nb-lstm-layers 1 --nb-lstm-units 64 --nb-conv-filters 16 --nb-dense-units 64 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-epochs 1 --early-stopping 10 --optimizer 'adam' --lr 0.001 --batch-size 10 --seq-length 20 --round-to-batch True --train-horses '[5,0,1,2]' --val-horses '[4]' --test-horses '[3]' --nb-workers 1 --device '/gpu:1' --image-identifier '1sttest_OF' --test-run 1 --nb-input-dims 5