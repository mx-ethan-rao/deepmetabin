
---

<div align="center">

# Metagenomic Binning

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://app.codiga.io/"><img alt="Code Grade" src="https://api.codiga.io/project/33753/status/svg"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
</div>

## Description

This repo contains implementations of four methods of metagenomic binning so-called GMAVE, DeepBin, GMGAT. Will integrate the GMGAT-Bert and vamb model as feature engineering components later. DeepBin and GMVAE shared same datamodule and lightningmoudle, only difference is reconstructing neighbrs in training_step.

The code repository is organized into the following components:
| Component | Description |
| --- | --- |
| [datamodules](https://github.com/eddiecong/Test-binning/tree/main/src/datamodules) | Contains torch dataset objects and pl.LightningDataModules for gmave, deepbin, pure gmgat methods. |
| [models](https://github.com/eddiecong/Test-binning/tree/main/src/models) | Contains torch module objects and pl.LightningModules for gmvae, deepbin, pure gmgat methos. |
| [utils](https://github.com/eddiecong/Test-binning/tree/main/src/utils) | Contains util functions in the project, shared visualization and evaluation functions across different model backbones. |
| [configs](https://github.com/eddiecong/Test-binning/tree/main/configs) | Contains hydra based config files to control the experiments across differernt models. |


## To Do List:
- :white_check_mark: warp up deepbin models.
- :white_check_mark: update GMGAT proposal models.
- :black_square_button: warp up graph bert models.
- :black_square_button: warp up repbin models.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/eddiecong/Metagenomic-Binning.git
cd Metagenomic-Binning

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python run.py experiment=train_gmgat.yaml
```

You can override any parameter from command line like this

```bash
python run.py trainer.max_epochs=20
```

## Preprocess Dataset From Scratch:
```
python preprocessing --outdir out --fasta my_contigs.fna --bamfiles *.bam
```

