# FSRB-Net

## Installation

```bash
python -m pip install -r requirements.txt
```

## 1. Data Preparation

The raw HUSTbearing and CWRU datasets are provided in this repository. Their original sources are:

- HUSTbearing: https://github.com/CHAOZHAO-1/HUSTbearing-dataset
- CWRU: https://engineering.case.edu/bearingdatacenter/download-data-file

The datasets are organized in `data/HUST/raw/` and `data/CWRU/raw/`, respectively. Check the datasets before generating images.

```bash
python -u run_prepare.py --dataset HUST --raw-dir ./data/HUST/raw --output-dir ./data/HUST/images/cwt --image-way cwt --scan-only

python -u run_prepare.py --dataset CWRU --raw-dir ./data/CWRU/raw --output-dir ./data/CWRU/images/cwt --image-way cwt --scan-only
```

## 2. CWT Image Generation

```bash
python -u run_prepare.py --dataset HUST --raw-dir ./data/HUST/raw --output-dir ./data/HUST/images/cwt --image-way cwt --skip-existing

python -u run_prepare.py --dataset CWRU --raw-dir ./data/CWRU/raw --output-dir ./data/CWRU/images/cwt --image-way cwt --skip-existing
```

## 3. Training

```bash
python -u run_train.py --dataset HUST --experiment custom --model fsrb_net --da dann --data-dir ./data/HUST/images/cwt --image-way cwt --output-dir ./outputs --seed 0 --epochs 30 --batch-size 40 --num-workers 4

python -u run_train.py --dataset CWRU --experiment custom --model fsrb_net --da dann --data-dir ./data/CWRU/images/cwt --image-way cwt --output-dir ./outputs --seed 0 --epochs 30 --batch-size 40 --num-workers 4
```

## 4. Evaluation

Evaluation is performed automatically after training each transfer task. Checkpoints, task-level metrics, and experiment summaries are saved under `outputs/`. Unlabeled target samples used for adaptation are evaluated after training (transductive UDA).

For additional options:

```bash
python run_prepare.py --help

python run_train.py --help
```
