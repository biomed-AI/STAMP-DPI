# STAMP-DPI Model
Deep Learning for DPI prediction

## Environment Requirement
cuda = 10.1

## Installation
It's easier to use this prediction script via Conda. Try to install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)

Use environment.yaml to setup fastly.
```bash
conda env create -f environment.yaml
```
Then install some third-party libraries

```bash
conda activate cpienv

conda install -c rmg descriptastorus
pip install pandas-flavor
pip install git+https://github.com.cnpmjs.org/samoturk/mol2vec

```

## Data
We use the dropbox restore the GalaxyDB and Davis dataset, which include the origianl drug-protein interaction data and calculated protein features.
The link is https://www.dropbox.com/sh/7maad34abz1knsp/AADfwkvm_Fu65vjtOwo5qcBNa?dl=0
After download the datasets, we assume that you unzip and place them int directory 'data' in our project.

## Evaluation from pretrained model.
We assume you are currently in the X-DPI project folder. If you want to evaluation the performance on our provided testing set, you can run the script named "run_test_v0.sh", such as
```
sh scripts/run_test_v0.sh
```

## Training and evaluating
if you want to train the X-DPI Model based on our provided training set, you should run the script named "run_train_v1.sh", such as
```
sh run_train_v1.sh
```
After training, we obtained the trained model, the we can use this model to evaluation on our testing set, such as
```
sh run_test_v1.sh
```

## Citation
If you find this code useful for your research, please use the following citation.
```
Penglei Wang†, Shuangjia Zheng†, Yize Jiang, Chengtao Li, Junhong Liu, Chang Wen, Atanas Patronov, Dahong Qian*, Hongming Chen* and Yuedong Yang*. Structure-aware multi-modal deep learning for drug-protein interactions prediction.
```
