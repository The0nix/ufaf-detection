# Uncertainty Estimation in LiDAR detection
![python3.8](https://img.shields.io/badge/python-v3.8-blue)

This is the repository for Deep Learning course final project made by Tamerlan Tabolov, Anton Semenkin, Natalia Soboleva and Aleksey Postnikov

## Installation
Create your virtual environment with Python 3.8+ and install dependencies using
```
pip install -r requirements.txt
```

## Usage
First you need to download and uncompress the nuScenes dataset: https://www.nuscenes.org/download.
You can download metadata and LiDAR sweeps only for the network to work.
### Training
To train the neural network you need first to create config as shown in [example config](./example-config.yaml)

Most importantly you should provide the version of your dataset as `nuscenes_version` (either `v1.0-trainval` or `v1.0-mini`) and the number of scenes you downloaded as `n_scenes`.

After that you can launch training with the following command
```
./main.py train -c path/to/config.yaml -d path/to/data/ -o path/to/model/saves/dir/ [-g GPU-LIST]
```

### Uncertainty estimation
For obtaining uncertainty estimates you have to edit [./mc_dropout.py](mc_dropout.py) and provide proper `data_path` and `model_path`, `n_scenes` and `version` variables as well as provide `data_number` of your choice indicating the index of a sample you want to evaluate on. After that you can launch
```
python mc_dropout.py
```
and resulting images will be saved into `pics/` directory
