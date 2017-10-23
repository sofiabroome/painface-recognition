# painface-recognition
Using convolutional LSTM networks to process videos of horses' facial pain expressions.

## Training

Look at the different shell-scripts available in root. A typical execution line may look like this

``` python main.py --data-path 'data/jpg_320_180_1fps/' --model 'convolutional_LSTM' --input-width 320 --input-height 180 --nb-labels 2 --nb-lstm-units 64 --nb-conv-filters 16 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch True --train-horses '[5,0,1,2]' --val-horses '[4]' --test-horses '[3]' --image-identifier 'jpg_val4_t3_seq10_1hl' --test-run 0 --seq-length 10 --nb-workers 1 --batch-size 2 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 0 ```, where

`--dropout-x`, `nb-lstm-units`, `nb-lstm-layers` and `nb-conv-filters` do not apply to all models, check `models.py`, `lr` is learning rate of the optimizer, `--round-to-batch` decides whether to discard the potential last batch if it has fewer samples than `--batch-size`, `train/val/test-horses` designate the ID of which horses to t/v/t on,  `--image-identifier` is a string to signify the results files that are saved after testing, to not overwrite results from different runs, `--test-run` is a binary variable deciding whether to just run a short test (just a few training steps instead of an entire epoch).

`--val-fraction` set to 1 means that we will use a certain last specified fraction of the training set as validation set, instead of using an "entire horse" as validation set as specified in `--val-horses` (applied otherwise).

`--nb-workers` is a GPU-setting that just handles 1 at the moment (need to implement different threads to use it).

`--nb-input-dims` is 5 for sequences but 4 for single frames (batch size, seq length, r, g, b) vs. (batch size, r, g, b)

`--data-type` can be either rgb or optical flow (of). The optical flow dimensions for sequences or single frames will also be 5 or 4, the workaround being to set the two last color channels to zero. 

## Testing

Run

```python test_with_saved_model.py```

and add the model path to its main function, to only do inference with an already trained keras model.
