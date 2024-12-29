import logging
import os
from typing import Dict, OrderedDict

import torch

from transformers import Trainer

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)

class BaseTrainer(Trainer):
    def __init__(self, *args, predict_dataset = None, test_key = "accuracy", gen_metric = None, gen_metric_key = None, gen_tokenizer = None, gen_dataset = None, gen_data_collator = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })
        self.gen_metric = gen_metric
        self.gen_metric_key = gen_metric_key
        self.gen_tokenizer = gen_tokenizer
        self.gen_dataset = gen_dataset
        self.gen_data_collator = gen_data_collator

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

    def evaluate(self, *args, **kwargs):
        eval_results = super().evaluate(*args, **kwargs)
        
        if self.gen_dataset is None: 
            return eval_results
        
        eval_dataloader = DataLoader(
            self.gen_dataset,
            sampler=self._get_eval_sampler(self.gen_dataset),
            batch_size=self.args.eval_batch_size*4,
            collate_fn=self.gen_data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
        references = []
        predictions = []
        
        self.model.eval()
        
        cnt = 0
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(self.model.device)
            # print(input_ids[:1])
            labels = batch["labels"].to(self.model.device) 
            attention_mask = batch["attention_mask"].to(self.model.device) 
            # print(input_ids.shape)
            # print(attention_mask.shape)
            # print(input_ids[:1])
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask, 
                max_new_tokens=70,
                num_return_sequences=1,
                do_sample=False, 
                # eos_id=self.gen_tokenizer.eos_token_id
            )
            # print(generated_ids[:1])
            input_prompts = [self.gen_tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids]
            generated_texts = [self.gen_tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]
            reference_texts = [
                self.gen_tokenizer.decode(label[label != -100], skip_special_tokens=True) for label in labels
            ]

            trimmed_generated_texts = []
            trimmed_reference_texts = []

            # print(f"input_prompts[0] = {input_prompts[0]}")
            # print(f"generated_texts[0] = {generated_texts[0]}")
            # print(f"reference_texts[0] = {reference_texts[0]}")

            for generated_text, reference_text, input_prompt in zip(generated_texts, reference_texts, input_prompts):
                if generated_text.startswith(input_prompt):
                    generated_text = generated_text[len(input_prompt):].strip()
                if reference_text.startswith(input_prompt):
                    reference_text = reference_text[len(input_prompt):].strip()

                trimmed_generated_texts.append(generated_text)
                trimmed_reference_texts.append(reference_text)
            # print("--final--")
            print(f"{trimmed_generated_texts[0][:10]}")
            # print(trimmed_reference_texts[:2])
            # input()
            
            predictions.extend(trimmed_generated_texts)
            references.extend(trimmed_reference_texts)
            cnt += 1
            if cnt > 10: break
        
        rouge_l_score = self.gen_metric(references=references, predictions=predictions)
        
        print(rouge_l_score)
        # input()
        
        eval_results[f"eval_{self.gen_metric_key}"] = rouge_l_score[self.gen_metric_key]
        
        return eval_results

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            print("in should_log")
            logs: Dict[str, float] = {}


            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            print(f"ignore_keys_for_eval = {ignore_keys_for_eval}")
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)
            print(f"cur: {eval_metrics}")

            if eval_metrics["eval_"+self.test_key] > self.best_metrics["best_eval_"+self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_"+self.test_key] = eval_metrics["eval_"+self.test_key]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics["test_"+self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_"+self.test_key] = test_metrics["test_"+self.test_key]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
