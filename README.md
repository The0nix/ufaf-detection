# Uncertainty Estimation in LiDAR detection
![python3.8](https://img.shields.io/badge/python-v3.8-blue)
![pytorch1.5.0](https://img.shields.io/badge/pytorch-v1.5.0-brightgreen)

This is the repository for Deep Learning course final project made by Tamerlan Tabolov, Anton Semenkin, Natalia Soboleva and Aleksey Postnikov.

Detection is made in the Single Shot Detector fashion, implementing early fusion model from [Fast & Furious paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf).
Uncertainty estimation is made via [Markov Chain Dropout](https://arxiv.org/abs/1506.02142) technique.

![results](https://i.ibb.co/SQktmFq/with-gt.png)

## Installation
Create your virtual environment with Python 3.8+ and install dependencies using
```
virtualenv .env --python=python3.8
. .env/bin/activate
pip install -r requirements.txt
```
Note that you will need Ubuntu 18.04 for simple installation. Otherwise it's needed to build open3d from source.

## Usage
First you need to download and uncompress the nuScenes dataset: https://www.nuscenes.org/download.
You can download metadata and LiDAR sweeps only for the network to work.

To perform any actions with the neural network you need first to create config as shown in [example config](./example-config.yaml).
Most importantly you should provide the version of your dataset as `nuscenes_version` (either `v1.0-trainval` or `v1.0-mini`), path to unarchived data as `data_path` and the number of scenes you downloaded as `n_scenes`.

When done you can use one of the following commands:
### Training
```
python ./main.py train -c path/to/config.yaml -o path/to/model/saves/dir/ [-g GPU-LIST -t tensorboard/logs/dir/]
```
### Validation
```
python ./main.py eval -c path/to/config.yaml -m path/to/model/checkpoint.pth
```

### Uncertainty estimation
```
python ./main.py mc-dropout -c path/to/config.yaml -m path/to/model/checkpoint.pth [-s path/to/plots/saves/dir/]
```
