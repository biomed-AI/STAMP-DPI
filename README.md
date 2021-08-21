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
We use the figshare restore the GalaxyDB and Davis dataset, which include the origianl drug-protein interaction data and calculated protein features. And the license is GPL 3.0+.

The link is:

Davis Dataset: https://doi.org/10.6084/m9.figshare.15169527.v1

GalaxyDB Dataset: https://doi.org/10.6084/m9.figshare.15169530

After download the datasets, we assume that you unzip and place them into directory 'data' in our project.

## Features
### Protein features
We calculated the features of proteins in GalaxyDB and Davis Dataset. For seen proteins in our dataset, the features can be find in the Data section. For unseen proteins in our dataset, you can calculate the features through the following software:

[Tape](https://github.com/songlab-cal/tape.git) for tape embedding.

[SPIDER3](https://sparks-lab.org/server/spider3/) for predicted structural features.

[PSI-BLAST v2.7.1](https://blast.ncbi.nlm.nih.gov/Blast.cgi) for PSSM features.

[HHBLITS v3.0.3](https://www.nature.com/articles/nmeth.1818) for HMM features.


### Compound featres
You can utlize the script (**src/preprocessing/compound.py**) to calculate the features for compounds. You need call the **get_mol2vec_features** and **get_mol_features** functions in the script to calculate the molecular representations used in the model building.

## Evaluation from pretrained model
We assume you are currently in the STAMP-DPI project folder. If you want to evaluation the performance on our provided testing set, you can run the command such as:
```
python src/main.py --mode test --root_data_path data/GalaxyDB --ckpt_path ckpts/GalaxyDB/final_model/final.ckpt --objective classification --pretrained 1
```

## Training and evaluating
If you want to train the STAMP-DPI Model based on our provided training set, you should run the command as follow
```
python src/main.py --mode train --root_data_path data/GalaxyDB --objective classification --gpus 0,1,2,3 --ckpt_save_path ckpts/GalaxyDB/
```
After training, we obtained the trained model, the we can use this model to evaluation on our testing set, you can run the command as follow (The parameter of ckpt_save_path is the path of checkpoint model):
```
python src/main.py --mode test --root_data_path data/GalaxyDB --ckpt_path ckpt_save_path --objective classification
```

## Training and evaluating for individual data
### Step 1
Prepared the dataset which includes the compound smiles, protein sequence and their interaction label. The dataset format and organization can refer our GalaxyDB or Davis dataset. At the same time, you need to pre-calculated the protein features as descirbed in the Features Section.
## Step 2
Training and evaluating the model. You need replace the "--root_data_path" with your own dataset path in Training and evaluating Section.


## Citation
If you find this code useful for your research, please use the following citation for our preprint paper.
```
X-DPI: A structure-aware multi-modal deep learning model for drug-protein interactions prediction. Penglei Wang, Shuangjia Zheng, Yize Jiang, Chengtao Li, Junhong Liu, Chang Wen, Atanas Patronov, Dahong Qian, Hongming Chen, Yuedong Yang. bioRxiv 2021.06.17.448780; doi: https://doi.org/10.1101/2021.06.17.448780
```
