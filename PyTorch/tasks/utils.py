# from tasks.glue.dataset import task_to_keys as glue_tasks
from tasks.superglue.dataset import task_to_keys as superglue_tasks

# GLUE_DATASETS = list(glue_tasks.keys())
SUPERGLUE_DATASETS = list(superglue_tasks.keys())
NER_DATASETS = ["conll2003", "conll2004", "ontonotes"]
SRL_DATASETS = ["conll2005", "conll2012"]
QA_DATASETS = ["squad", "squad_v2"]
ALPACA_DATASETS = ["alpaca"]

TASKS = ["superglue", "ner", "srl", "qa", "alpaca"]

DATASETS = SUPERGLUE_DATASETS + NER_DATASETS + SRL_DATASETS + QA_DATASETS + ALPACA_DATASETS

ADD_PREFIX_SPACE = {
    'bert': False,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': True,
}

USE_FAST = {
    'bert': True,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': False,
}