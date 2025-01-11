## News 

- **[2024/11/10]** Closed-loop code for [GenAD](https://github.com/wzzheng/GenAD) has been released.

## Demo

![vis](../assets/carla.png)

**Comparisons of the proposed generative end-to-end autonomous driving framework with the conventional pipeline.** Bench2Drive comprises the [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) repository for closed-loop evaluation and the model repository [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/uniad/vad). The code in this repository integrates GenAD within the Bench2DriveZoo repository, with the majority of the code being identical to that in Bench2DriveZoo. This repository does not contain the code from the Bench2Drive repository, and no modifications were made to the closed-loop evaluation code. Only the execution scripts were adjusted, as detailed in the following description.

## Results

|       Method        | Driving Score | Success Rates (%) |
| :-----------------: | :-----------: | :---------------: |
|     VAD (Paper)     |     39.42     |        0.1        |
| VAD (Github Update) |     42.35     |       0.13        |
| VAD (Reproduction)  |     38.16     |       0.15        |
|      **GenAD**      |   **44.81**   |     **0.159**     |

## Getting Started
### 0.installation

Clone this repository and configure it according to the *Getting Started* section in the [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/uniad/vad) repository documentation. Refer to the configuration documentation in the [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)  repository to link this repository to the closed-loop evaluation repository.

Detailed package versions can be found in [requirements.txt](../requirements.txt).


### 1.Training

``` 
sh ./adzoo/genad/dist_train.sh ./adzoo/genad/configs/VAD/GenAD_config_b2d.py 1
```

**Note:** Detailed training and evaluation methods can be found in the documentation of [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/uniad/vad).

### 2.Open-Loop Evaluation

```
sh ./adzoo/genad/dist_test.sh ./adzoo/genad/configs/VAD/GenAD_config_b2d.py ./work_dirs/GenAD_config_b2d/epoch_.pth 1
```

### 3.Closed-Loop Evaluation

Eval GenAD with 8 GPUs

```shell
leaderboard/scripts/run_evaluation_multi.sh
```

Eval GenAD with 1 GPU

```shell
leaderboard/scripts/run_evaluation_debug.sh
```

**Note:** Detailed training and evaluation methods can be found in the documentation of [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/uniad/vad).

## Acknowledgement
[VAD](https://github.com/hustvl/VAD),
[UniAD](https://github.com/OpenDriveLab/UniAD),
[GenAD](https://github.com/wzzheng/GenAD),
[Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo)

## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{zheng2024genad,
    title={GenAD: Generative End-to-End Autonomous Driving},
    author={Zheng, Wenzhao and Song, Ruiqi and Guo, Xianda and Zhang, Chenming and Chen, Long},
    journal={arXiv preprint arXiv: 2402.11502},
    year={2024}
}
```

