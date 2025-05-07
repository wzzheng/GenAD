# Prepare Bench2Drive Dataset

## Download Bench2Drive

Download our dataset from ([LINK](https://github.com/Thinklab-SJTU/Bench2Drive)) and make sure the structure of data as follows:

**Notice: some version of data may have slightly different folder structure. You may need to use soft link (ln -s) and change the path related code.**

```
    Bench2DriveZoo
    ├── ...                   
    ├── data/
    |   ├── bench2drive/
    |   |   ├── v1/                                          # Bench2Drive base 
    |   |   |   ├── Accident_Town03_Route101_Weather23/
    |   |   |   ├── Accident_Town03_Route102_Weather20/
    |   |   |   └── ...
    |   |   └── maps/                                        # maps of Towns
    |   |       ├── Town01_HD_map.npz
    |   |       ├── Town02_HD_map.npz
    |   |       └── ...
    |   ├── others
    |   |       └── b2d_motion_anchor_infos_mode6.pkl        # motion anchors for UniAD
    |   └── splits
    |           └── bench2drive_base_train_val_split.json    # trainval_split of Bench2Drive base 

```

## Prepare Bench2Drive data info

Run the following command:

```
cd mmcv/datasets
python prepare_B2D.py --workers 16   # workers used to prepare data
```

The command will generate `b2d_infos_train.pkl`, `b2d_infos_val.pkl`, `b2d_map_infos.pkl` under `data/infos`.

*Note: This command will be by default use all routes except those in data/splits/bench2drive_base_train_val_split.json as the training set.  It will take about 1 hour to generate all the data with 16 workers for Base set (1000 clips).*


## Structure of code


After installing and data preparing, the structure of our code will be as follows:

```
    Bench2DriveZoo
    ├── adzoo/
    |   ├── bevformer/
    |   ├── uniad/
    |   └── vad/                   
    ├── ckpts/
    |   ├── r101_dcn_fcos3d_pretrain.pth                   # pretrain weights for bevformer
    |   ├── resnet50-19c8e357.pth                          # image backbone pretrain weights for vad
    |   ├── bevformer_base_b2d.pth                         # download weights you need
    |   ├── uniad_base_b2d.pth                             # download weights you need
    |   └── ...
    ├── data/
    |   ├── bench2drive/
    |   |   ├── v1/                                        # Bench2Drive base 
    |   |   |   ├── Accident_Town03_Route101_Weather23/
    |   |   |   ├── Accident_Town03_Route102_Weather20/
    |   |   |   └── ...
    |   |   └── maps/                                      # maps of Towns
    |   |       ├── Town01_HD_map.npz
    |   |       ├── Town02_HD_map.npz
    |   |       └── ...
    │   ├── infos/
    │   │   ├── b2d_infos_train.pkl
    │   │   ├── b2d_infos_val.pkl
    |   |   └── b2d_map_infos.pkl
    |   ├── others
    |   |       └── b2d_motion_anchor_infos_mode6.pkl      # motion anchors for UniAD
    |   └── splits
    |           └── bench2drive_base_train_val_split.json  # trainval_split of Bench2Drive base 
    ├── docs/
    ├── mmcv/
    ├── team_code/  # for Closed-loop Evaluation in CARLA
```



