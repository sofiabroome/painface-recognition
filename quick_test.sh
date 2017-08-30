#!/bin/bash
python main.py --data-path 'data_test/' --model 'conv2d_lstm_stateful' --input-width 320 --input-height 180 --nb-labels 2 --nb-lstm-units 5 --nb-conv-filters 5 --kernel-size 5 --dropout-rate 0.5 --nb-epochs 3 --early-stopping 50 --optimizer 'adam' --lr 0.01 --round-to-batch True --device '/gpu:1' --image-identifier 'test'
