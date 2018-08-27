# painface-recognition
Using convolutional LSTM networks (or other models) to classify videos of horses' pain (or no pain) expressions.

## Training

Look at the different shell-scripts available in `run_scripts` and `test_scripts` folders. An execution line with all flags specified may look like this

```python main.py --data-path 'data/jpg_128_128_2fps/' --model 'convolutional_LSTM' --input-width 128 --input-height 128 --nb-labels 2 --nb-lstm-units 32 --nb-conv-filters 16 --kernel-size 5 --dropout-1 0.25 --dropout-2 0.5 --nb-epochs 100 --early-stopping 15 --optimizer 'adadelta' --lr 0.001 --round-to-batch 1 --train-horses '[3,0,1,2]' --val-horses '[4]' --test-horses '[5]' --image-identifier 'unique_name' --test-run 0 --seq-length 10 --seq-stride 10 --nb-workers 1 --batch-size 16 --nb-input-dims 5 --val-fraction 0 --data-type 'rgb' --nb-lstm-layers 4 --aug-flip 1 --aug-crop 1 --aug-light 1 ```, where

`--dropout-x`, `nb-lstm-units`, `nb-lstm-layers` and `nb-conv-filters` do not apply to all models, check `models.py`, `lr` is learning rate of the optimizer, `--round-to-batch` decides whether to discard the potential last batch if it has fewer samples than `--batch-size`, `train/val/test-horses` designate the ID of which horses to t/v/t on,  `--image-identifier` is a string to signify the results files that are saved after testing, to not overwrite results from different runs, `--test-run` is a binary variable deciding whether to just run a short test (just a few training steps instead of an entire epoch). `aug-flip`, `aug-crop` and `aug-light` are three different data augmentation methods that can be used all at will. The batch size needs to be divisible with the number of augmentation techniques you use plus + 1. So if you use all three, i.e. flip+crop+light, the batch size needs to be a multiple of 4.

`--val-fraction` set to 1 means that we will use a certain last specified fraction of the training set as validation set, instead of using an "entire horse" as validation set as specified in `--val-horses` (applied otherwise).

`--nb-workers` more than 1 is not supported.

`--nb-input-dims` is 5 for sequences but 4 for single frames.
In the sequence case the dimensions of an input tensor to the network are (batch size, seq length, r, g, b) vs. in the single-frame case (batch size, r, g, b).

`--data-type` can be either rgb or optical flow (of). The optical flow dimensions for sequences or single frames will also be 5 or 4, the workaround being to set the two last color channels to zero. 

`seq-length` is the sequence length (number of frames to constitute a sequence that is processed at once, and `sseq-stride` is the stride of the sequence extraction. If you just want back-to-back sequences with no overlap these two should thus be the same. 

## Testing

To only perform inference with an already trained Keras model, run

```python test_with_saved_model.py```

with the same argument flags as you trained the model with

and add the model path to the main function of `test_with_saved_model.py`.

## Data format

A data folder such as the one in the execution line specified above, `data/jpg_128_128_2fps`, (let's refer to this folder by X) should be organized as follows:

 X contains one subfolder per subject in the dataset (and once you have run, also their belonging `.csv`-files), like so:
 
```
ls X
horse_1 horse 2 horse_3 horse_4 horse_5 horse_6
```

and after the first run:

```
ls X
horse_1 horse_1.csv horse 2 horse_2.csv horse_3 horse_3.csv horse_4 horse_4.csv horse_5 horse_5.csv horse_6 horse_6.csv 
```

These per-subject folders in turn contain one folder per clip, as in for example:

```
ls horse_1
1_1a_1 1_1a_2 1_1a_3 1_1b   1_2a   1_2b   1_3a_1 1_3a_2 1_3b   1_4    1_5_1  1_5_2
```


