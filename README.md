# painface-recognition
Using convolutional LSTM networks to process videos of horses' facial pain expressions.

## Training

Look at the different shell-scripts available in root. A typical execution line may look like this

``` python main.py --data-path 'data/jpg_320_180_1fps/' --model 'convolutional_LSTM' --input-width 320 --input-height 180 --nb-labels 2 --nb-lstm-units 64 --nb-conv-filters 16 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch True --train-horses '[5,0,1,2]' --val-horses '[4]' --test-horses '[3]' --device '/gpu:1' --image-identifier 'jpg_val4_t3_seq10_1hl' --test-run 0 --seq-length 10 --nb-workers 1 --batch-size 2 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 0 ```

## Testing


