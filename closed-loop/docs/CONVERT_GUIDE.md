# Code Convert Guide

This document outlines important considerations for migrating code based on nuscenes or other datasets to bench2drive.

## Models

We integrated several MMCV dependencies into the `mmcv` directory and no longer install the original libraries. You can refer to our existing methods to utilize these modules and place your own models and utils in `mmcv` directory and register them. Please make sure the mmcv directory contains all the modules you need; if not, you will need to add them.

## Scripts and configs

You can place the configs and scripts for each method in the `adzoo` . Utils of each methods can also be placed here for easier management.

## Details of configs

To create a config for the bench2drive dataset, note the following:

- We have included the bench2drive name-to-class mapping and evaluation settings directly in the config. You can use our settings or modify them as needed.
- Unlike the 10 classes in nuscenes, we use 9 classes in bench2drive .
- Methods like UniAD and VAD use 3 commands on nuscenes, while bench2drive uses 6 commands obtained from CARLA.

## Dataset

- The coordinate system of the Bench2Drive data differs significantly from the coordinate system used by BEVFormer/UniAD/VAD.([here](https://github.com/Thinklab-SJTU/Bench2Drive/blob/main/docs/anno.md) for details). In `mmcv/datasets/prepare_B2D.py`, we convert the world coordinate system, ego coordinate system, and sensor coordinate system to match their coordinate system, including the vehicle coordinates, bounding box coordinates, and sensor extrinsics. You can refer to our code for data alignment. 
- In Nuscenes, keyframes are at 2Hz, while Bench2Drive runs at 10Hz with annotations for each frame. For reproducing UniAD and VAD, we set the window length (time interval between adjacent points in past and future trajectories) to 0.5s and the window shift to 0.1s (any frame can be selected as the current frame). This fully utilizes Bench2Drive's data and aligns the trajectories with Nuscenes.
- For the map, Bench2Drive stores vectorized maps.  You can refer to our code to use the map, such as extracting map elements within a certain range.

## Team agent

To perform closed-loop evaluation in CARLA, set up sensors to gather data from CARLA. Use this data to compute all necessary model inputs, then convert the model outputs into a `carla.VehicleControl` object.
