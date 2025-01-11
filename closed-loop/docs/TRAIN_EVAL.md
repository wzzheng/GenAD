# Train/Eval models

You can use following commands to train and validate [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [UniAD](https://github.com/OpenDriveLab/UniAD) and [VAD](https://github.com/hustvl/VAD)

## BEVFormer

### Train 

```bash
#train BEVFormer base
./adzoo/bevformer/dist_train.sh ./adzoo/bevformer/configs/bevformer/bevformer_base_b2d.py 4 #N_GPUS
#train BEVFormer tiny
./adzoo/bevformer/dist_train.sh ./adzoo/bevformer/configs/bevformer/bevformer_tiny_b2d.py 4 #N_GPUS
```
### Open loop eval 

```bash
#eval BEVFormer base
./adzoo/bevformer/dist_test.sh ./adzoo/bevformer/configs/bevformer/bevformer_base_b2d.py ./ckpts/bevformer_base_b2d.pth 1 
#test BEVFormerr tiny
./adzoo/bevformer/dist_test.sh ./adzoo/bevformer/configs/bevformer/bevformer_tiny_b2d.py ./ckpts/bevformer_tiny_b2d.pth 1 
```


## UniAD

### Train stage1
```bash
#train UniAD base
./adzoo/uniad/uniad_dist_train.sh  ./adzoo/uniad/configs/stage1_track_map/base_track_map_b2d.py 4 
#train UniAD tiny
./adzoo/uniad/uniad_dist_train.sh  ./adzoo/uniad/configs/stage1_track_map/tiny_track_map_b2d.py 4 
```

### Train stage2
```bash
#train UniAD base
./adzoo/uniad/uniad_dist_train.sh  ./adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py 1 
#train UniAD tiny
./adzoo/uniad/uniad_dist_train.sh  ./adzoo/uniad/configs/stage2_e2e/tiny_e2e_b2d.py 1 
```


### Open loop eval

```bash
#eval UniAD base
./adzoo/uniad/uniad_dist_eval.sh ./adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py ./ckpts/uniad_base_b2d.pth 1
#eval UniAD tiny
./adzoo/uniad/uniad_dist_eval.sh ./adzoo/uniad/configs/stage2_e2e/tiny_e2e_b2d.py ./ckpts/uniad_tiny_b2d.pth 1
```


## VAD

### Train 

```bash
./adzoo/vad/dist_train.sh ./adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py ./ckpts/vad_b2d_base.pth 1
```

### Open loop eval

```bash
./adzoo/vad/dist_test.sh ./adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py ./ckpts/vad_b2d_base.pth 1
```

**NOTE**: UniAD and VAD use different definitions to calculate Planning L2. UniAD calculates L2 at each time step(0.5s,1.0s,1.5s,...), while VAD calculates the average over each time period(0s-0.5s,0s-1.0s,0s-1.5s,...). We retain the original calculation logic in the code, but report UniAD's Planning L2 converted to VAD's definition.
