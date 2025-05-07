# Closed Loop Evaluation    

Please follow these steps to evaluate UniAD and VAD in CARLA:

## Preparations

- Install this repo as [doc](docs/INSTALL.md). 
- Clone Bench2Drive evaluation tools from [here](https://github.com/Thinklab-SJTU/Bench2Drive) and prepare CARLA For it.

## Link this repo to Bench2Drive

```bash
# Add your agent code
cd Bench2Drive/leaderboard
mkdir team_code
ln -s Bench2DriveZoo/team_code/* ./team_code    # link UniAD,VAD agents and utils 
cd ..
ln -s Bench2DriveZoo  ./                        # link entire repo to Bench2Drive. 
```

## Run evaluation 

Follow [this](https://github.com/Thinklab-SJTU/Bench2Drive?tab=readme-ov-file#eval-tools) to use evaluation tools of Bench2Drive.

