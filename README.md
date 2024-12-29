# Reproduction of P-tuning-v2 on RoBERTa, GLM and GPT

This is the reproduction version readme, modified from the original version. 

This folder contains the PyTorch & Jittor version of the reproduction code. 

## Model Weight

You should download GLM-2b manually from [huggingface-glm-2b](https://huggingface.co/THUDM/glm-2b/resolve/main/pytorch_model.bin) and put it at `PyTorch/tasks/glm-2b/pytorch_model.bin` and `Jittor/tasks/glm-2b/pytorch_model.bin` . 

## PyTorch Version

First, you should enter the folder `PyTorch/`. 

### Setup

To setup environment, you should: 

```shell
conda create -n final_pytorch python=3.8
conda activate final_pytorch
pip install torch==2.4.1 numpy==1.24.4 transformers==4.26.1 nltk==3.9.1 rouge-score==0.1.2 datasets==2.3.2 scikit-learn==1.3.2
```

### Training

For example, you shall run: 

```shell
bash run_script/run_roberta_pt2/run_boolq_roberta.sh
```

## Jittor



## Note

Other tasks are disabled, because we did not impletement them in Jittor. 

We use [glm-2b](https://github.com/THUDM/GLM) and [alpaca](https://github.com/tatsu-lab/stanford_alpaca) in this repository. 

You can refer to the original README in `PyTorch/README.md` for more info. 