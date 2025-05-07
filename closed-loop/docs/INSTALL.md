## Follow these steps to install the environment
- **STEP 1: Create enviroment**
    ```
    conda create -n b2d_zoo python=3.8
    conda activate b2d_zoo
    ```
- **STEP 2: Install cudatoolkit**
    ```
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    ```
- **STEP 3: Install torch**
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
- **STEP 4: Set environment variables**
    ```
    export PATH=YOUR_GCC_PATH/bin:$PATH
    export CUDA_HOME=YOUR_CUDA_PATH/
    ```
- **STEP 5: Install ninja and packaging**
    ```
    pip install ninja packaging
    ```
- **STEP 6: Install our repo**
    ```
    pip install -v -e .
    ```

- **STEP 7: Prepare pretrained weights.**
    create directory `ckpts`

    ```
    mkdir ckpts 
    ```
    Download `resnet50-19c8e357.pth` form [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/resnet50-19c8e357.pth) or [Baidu Cloud](https://pan.baidu.com/s/1LlSrbYvghnv3lOlX1uLU5g?pwd=1234 ) or from Pytorch official website.
  
    Download `r101_dcn_fcos3d_pretrain.pth` form [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/r101_dcn_fcos3d_pretrain.pth) or [Baidu Cloud](https://pan.baidu.com/s/1o7owaQ5G66xqq2S0TldwXQ?pwd=1234) or from BEVFormer official repo.


- **STEP 8: Install CARLA for closed-loop evaluation.**

    ```
    ## Ignore the line about downloading and extracting CARLA if you have already done so.
    mkdir carla
    cd carla
    wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
    tar -xvf CARLA_0.9.15.tar.gz
    cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
    cd .. && bash ImportAssets.sh
    export CARLA_ROOT=YOUR_CARLA_PATH

    ## Important!!! Otherwise, the python environment can not find carla package
    echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.7/site-packages/carla.pth # python 3.8 also works well, please set YOUR_CONDA_PATH and YOUR_CONDA_ENV_NAME

    ```
