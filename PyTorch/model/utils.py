from enum import Enum

from model.sequence_classification import (
    RobertaPrefixForSequenceClassification,
    RobertaPromptForSequenceClassification,
    GLMPrefixForSequenceClassification,
    GLMPromptForSequenceClassification
)

from model.multiple_choice import (
    RobertaPrefixForMultipleChoice,
    RobertaPromptForMultipleChoice,
    GLMPrefixForMultipleChoice
)

from model.causal_lm import (
    GLMPrefixForCausalLM, 
    RobertaPrefixForCausalLM, 
    GPT2PrefixLMHeadModel
)

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice
)

from model.glm2b.modeling_glm import GLMForConditionalGeneration

from transformers.models.roberta.modeling_roberta import RobertaForCausalLM

from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

class TaskType(Enum):
    TOKEN_CLASSIFICATION = 1,
    SEQUENCE_CLASSIFICATION = 2,
    QUESTION_ANSWERING = 3,
    MULTIPLE_CHOICE = 4, 
    CAUSAL_LM = 5

PREFIX_MODELS = {
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: RobertaPrefixForMultipleChoice,
        TaskType.CAUSAL_LM: RobertaPrefixForCausalLM
    },
    "glm": {
        TaskType.SEQUENCE_CLASSIFICATION: GLMPrefixForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: GLMPrefixForMultipleChoice, 
        TaskType.CAUSAL_LM: GLMPrefixForCausalLM
    }, 
    "gpt2": {
        TaskType.CAUSAL_LM: GPT2PrefixLMHeadModel
    }
}

PROMPT_MODELS = {
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPromptForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: RobertaPromptForMultipleChoice
    },
    "glm": {
        TaskType.SEQUENCE_CLASSIFICATION: GLMPromptForSequenceClassification,
    }
}

AUTO_MODELS = {
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
        TaskType.CAUSAL_LM: RobertaForCausalLM
    },
    "glm": {
        TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice, 
        TaskType.CAUSAL_LM: GLMForConditionalGeneration
    }, 
    "gpt2": {
        TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice, 
        TaskType.CAUSAL_LM: GPT2LMHeadModel
    }
    # TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    # TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    # TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    # TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
}

def get_model(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size
        
        model_class = PREFIX_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    elif model_args.prompt:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        model_class = PROMPT_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    else:
        model_class = AUTO_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )

        bert_param = 0
        if fix_bert:
            if config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model