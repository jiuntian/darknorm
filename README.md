# DarkNorm
DarkNorm, a simple dark standardization and Video TrivialAugment is all you need for 
hyperparameter-less action recognition in dark.

This repository is modeling on the foundation of DarkLight pytorch code. Thanks the author(s) for their great work.

## Dependencies

The code runs on Python 3.8 but in fact it is not a big deal to try to construct a low version Python. You can create a conda environment with all the dependecies by running 

```bash
conda env create -f requirements.yml -n darknorm
```

Note: this project needs the CUDA 11.3

## Dataset Preparation

In order to improve the dataset's load speed an facilitate pretreatment. The video should be pre-cut into frames and save in \datasets folder and the list of training and validation samples should be created as txt files in \datasets\settings folder.

For example, using ARID as the data. the frames is saved in \datasets\ARID_frames. And then using the different name of classifications to name the folder. 

The format of the frames like as "img_%05d.jpg" 

## Training

### Training with tow view

Reproducing training results

We run the experiments with two NVIDIA V100 16GB GPU. Please use the same setting to reproduce the exact result.

You should get best top 1 validation accuracy at 72.187 and with minimum loss of 0.9719.
```
CUDA_VISIBLE_DEVICES=0,1 python darknorm.py --batch-size=12 --workers 16 --arch DarkNorm --lr 0.0001 --tag triv_norm_8888 --seed 8888
```

To continue the training from the best model, add -c. To evaluate the single clip single crop performance of best model, add -e

## Testing

```
python spatial_demo_bert.py  --split=1
```

## Related Projects
We thank author of Darklight for the base of this repo.
[Darklight](https://github.com/Ticuby/Darklight-Pytorch): Darklight-Pytorch

[LateTemporalModeling3DCNN](https://github.com/artest08/LateTemporalModeling3DCNN):2_Plus_1D_BERT





