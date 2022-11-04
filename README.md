# DarkNorm
DarkNorm, a simple dark standardization and Video TrivialAugment is all you need for 
hyperparameter-less action recognition in dark.
To ensure fully reproducibility, this repository uses deterministic operations and user-specified seed.

This repository is modeling on the foundation of DarkLight pytorch code. Thanks the author(s) for their great work.

## Dependencies

The code runs on Python 3.8. You can create a conda environment with all the dependencies by running 

```bash
git clone https://github.com/jiuntian/darknorm.git
cd darknorm
conda env create -f requirements.yml -n darknorm
conda activate darknorm
```

Note: this project needs the CUDA 11.3

## Dataset Preparation

The video should be pre-cut into frames and save in `datasets` folder and the list of training
and validation samples should be created as txt files in `datasets/settings` folder.
Please follow the following instructions to prepare thed datasets.

1. First, download the provided datasets at 
https://drive.google.com/file/d/17kcKUailyLd8Pb2VOUMRrAlmfnVKyioN/view?usp=sharing.
2. Extract the datasets in folder `datasets/ee6222`. There supposed to be 3 files and two folders in `datasets/ee6222`: 
`mapping_table.txt  train  train.txt  validate  validate.txt`
3. Suppose that you are now at the project root, execute the following script to extract the video frames:
   ```bash
   sed -i '0,/^/s//VideoID\tClassID\tVideo\n/' datasets/ee6222/train.txt
   sed -i '0,/^/s//VideoID\tClassID\tVideo\n/' datasets/ee6222/validate.txt
   python utils/video_to_frame.py
   python utils/video_to_frame_val.py
   cp val_split1.txt datasets/settings/EE6222/val_split1.txt
   cp train_split1.txt datasets/settings/EE6222/train_split1.txt
   ```
   The frames should have extracted to `datasets/EE6222_frames`.
4. Preparing the labels by running: 
    ```bash
   python utils/csv_to_split.py
    ```


## Training

Reproducing training results:

We run the experiments with two NVIDIA V100 16GB GPU. Please use the same setting to reproduce the exact result.

You should get best top 1 validation accuracy at 72.187 and with minimum loss of 0.9719.
```
CUDA_VISIBLE_DEVICES=0,1 python darknorm.py --batch-size=12 --workers 16 --arch DarkNorm --lr 0.0001 --tag triv_norm_1234 --seed 1234
```

## Reproduce Testing Results
Test Video is available at [HERE](https://entuedu-my.sharepoint.com/:u:/g/personal/jiuntian001_e_ntu_edu_sg/EeXx_q612BhIsKT6_KISY0gBznyU3g60iQJb_--qrXIb0w?e=i8JnAe).
Ensure that the test videos is in `datasets/ee6222/test`.
The frames will be extracted to `datasets/EE6222_frames_test`.
Download trained model at 
[HERE](https://entuedu-my.sharepoint.com/:u:/g/personal/jiuntian001_e_ntu_edu_sg/EcMFXQ2p48xJnGB6g-o4PFIBvxc3EttxIs9Z5n27oMGNqw?e=53NGaM) 
and extract at root folder. There would be some trained checkpoints in `checkpoints` folder.
```bash
# extract frames
python utils/video_to_frame_test.py
# extract labels
sed -i '0,/^/s//VideoID\tClassID\tVideo\n/' datasets/ee6222/test.txt
python utils/csv_to_split_test.py
cp test_split1.txt datasets/settings/EE6222/test_split1.txt
# to evaluate
python eval.py --gpu 0 --ckpt checkpoints/gamma_ce_EE6222_DarkNormr18_split1_triv_norm_reprod_1234
```

## Related Projects
We thank author of Darklight for the base of this repo.
1. [Darklight](https://github.com/Ticuby/Darklight-Pytorch): Darklight-Pytorch
2. [LateTemporalModeling3DCNN](https://github.com/artest08/LateTemporalModeling3DCNN):2_Plus_1D_BERT
