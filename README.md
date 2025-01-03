# Reproduction of P-tuning-v2 on RoBERTa, GLM and GPT

This is the reproduction version of [P-tuning-v2](https://arxiv.org/abs/2110.07602), modified from the [official code](https://github.com/THUDM/P-tuning-v2). 

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

First, you should enter the folder `Jittor/`. 

### Setup

这一块尚在施工中。我们正在努力写出一个环境配置指南，包含(1)环境配置步骤，以及(2)对我们对环境做的改动的具体解释。

这份环境配置指南也将发布到 Jittor 社区中。目前发布在了 [Jittor 社区](https://discuss.jittor.org/t/topic/905) 

update:

已经在库中附带了 jtorch 和 transformers 的改动后版本，直接执行如下指令：

```shell
conda create -n final_jittor python=3.8
conda activate final_jittor
pip install jtorch_modified/. transformers_jittor_modified/.
pip install datasets==2.3.2 scikit-learn==1.3.2
```

### Training

For example, you shall run:

```shell
bash run_script/run_roberta_pt2/run_boolq_roberta.sh
```

You'll probably see `jit_utils updated, please rerun your command.` when you run the script for the first time, just ignore it and rerun. 

## Note

Other tasks are disabled, because we did not impletement them in Jittor. 

We use [glm-2b](https://github.com/THUDM/GLM) and [alpaca](https://github.com/tatsu-lab/stanford_alpaca) in this repository. 

You can refer to the original README in `PyTorch/README.md` for more info. 