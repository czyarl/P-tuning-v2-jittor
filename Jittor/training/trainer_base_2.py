import logging

import pdb

logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)

import torch

if torch.__version__ != '2.0.0':
    from transformers import AdamW
else:
    from jittor.nn import AdamW

import json
import os

from typing import Dict, Optional

from torch.utils.data import DataLoader, RandomSampler
from collections import OrderedDict
from transformers import TrainingArguments, EvalPrediction
from tqdm import tqdm

class BaseTrainer:
    def __init__(self, model, args: TrainingArguments, tokenizer, optimizer=None, lr_scheduler=None, 
                 train_dataset=None, eval_dataset=None, test_dataset=None, 
                 eval_metric=None, test_key="accuracy", gen_metric=None, 
                 gen_metric_key=None, gen_tokenizer=None, gen_dataset=None, 
                 gen_data_collator=None, data_collator=None, compute_metrics=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args  # TrainingArguments

        ## xhb
        if torch.__version__ != '2.0.0':
            self.optimizer = optimizer or AdamW(model.parameters(), lr=args.learning_rate)
        else:
            self.optimizer = optimizer or AdamW(list(model.parameters()), lr=args.learning_rate)
        model.float32()
        # for param in model.parameters():
        #     if param.dtype == torch.float16:
        #         print(f"param.dtype = {param.dtype}!!!WAWAWWAAWAWWAAWAW")
        print('----BaseTrainer-------')
        print(type(model.parameters()))
        print(type(model))
        # tmp = list(model.parameters())
        # print(type(tmp))
        # print(tmp, file = open('tmp1.txt', 'w'))
        print('----BaseTrainer-------')

        self.lr_scheduler = lr_scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.eval_metric = eval_metric
        self.test_key = test_key
        self.gen_metric = gen_metric
        self.gen_metric_key = gen_metric_key
        self.gen_tokenizer = gen_tokenizer
        self.gen_dataset = gen_dataset
        self.gen_data_collator = gen_data_collator
        self.data_collator = data_collator  # For general datasets (train/eval)
        self.compute_metrics = compute_metrics

        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })
    
    def evaluate(self, ignore_keys_for_eval=None):
        # 初始化评估结果字典
        eval_results = {}

        # 设置评估数据加载器
        eval_dataloader = DataLoader(
            self.eval_dataset,
            # sampler=RandomSampler(self.eval_dataset),
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            # drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            # pin_memory=self.args.dataloader_pin_memory,
        )

        # 初始化用来保存参考标签和预测标签
        references = []
        predictions = []

        # 将模型设置为评估模式
        self.model.eval()

        # 计数器，用于限制评估的批次数
        cnt = 0
        total_loss = 0.0
        total_batches = 0

        # 遍历评估数据集中的每个批次
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(self.model.device)
            labels = batch["labels"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            token_type_ids = None
            if "token_type_ids" in batch: 
                # print("token_type_ids")
                token_type_ids = batch["token_type_ids"].to(self.device)

            # 使用模型计算输出，并获取损失值
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=token_type_ids)
                loss = outputs.loss
                logits = outputs.logits

            total_loss += loss.item()
            total_batches += 1

            # 获取预测的标签，这里使用 argmax 来获取每个样本的预测类别
            predicted_labels = logits.argmax(dim=-1)  # 取最大概率的标签

            # 将真实标签和预测标签添加到列表中
            references.extend(labels.cpu().numpy())  # 转换为 CPU 和 numpy 数组
            predictions.extend(logits.cpu().numpy())  # 转换为 CPU 和 numpy 数组

        # 计算平均损失
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        eval_results["eval_loss"] = avg_loss

        # 如果有指定的 metric 函数，计算并添加到评估结果中（如 accuracy, f1 等）
        if self.compute_metrics is not None:
            eval_prediction = EvalPrediction(predictions=predictions, label_ids=references)
            metrics = self.compute_metrics(eval_prediction)  # 传递 EvalPrediction 对象
            eval_results.update({f"eval_{key}": value for key, value in metrics.items()})
            print(f"111: {eval_results}")


        if self.gen_dataset is None: 
            return eval_results
        
        eval_dataloader = DataLoader(
            self.gen_dataset,
            # sampler=RandomSampler(self.gen_dataset),
            batch_size=self.args.eval_batch_size*4,
            collate_fn=self.gen_data_collator,
            # drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            # pin_memory=self.args.dataloader_pin_memory,
        )
        
        references = []
        predictions = []
        
        self.model.eval()
        
        cnt = 0
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(self.model.device)
            labels = batch["labels"].to(self.model.device) 
            attention_mask = batch["attention_mask"].to(self.model.device) 
            token_type_ids = None
            if "token_type_ids" in batch: 
                # print("token_type_ids")
                token_type_ids = batch["token_type_ids"].to(self.device)
            # print(input_ids.shape)
            # print(attention_mask.shape)
            # print(input_ids[:1])
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
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
    
    def save_model(self, checkpoint_path):
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

    def log_metrics(self, prefix, metrics):
        """Log metrics to console or file."""
        logger.info(f"Logging {prefix} metrics: {metrics}")
        for key, value in metrics.items():
            logger.info(f"{prefix}_{key} = {value}")

    def save_metrics(self, prefix, metrics):
        """Save metrics to a JSON file."""
        metrics_file = os.path.join(self.args.output_dir, f"{prefix}_metrics.json")
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved {prefix} metrics to {metrics_file}")

    def save_state(self):
        """Save training state, including optimizer and scheduler."""
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "epoch": self.best_metrics["best_epoch"],
        }

        # Save the state to a file
        state_file = os.path.join(self.args.output_dir, "trainer_state.pth")
        torch.save(state, state_file)
        logger.info(f"Saved training state to {state_file}")

    def log_best_metrics(self):
        logger.info(f"Best metrics: {self.best_metrics}")
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        # Log training loss and learning rate
        # if self.control.should_log:
        #     logs: Dict[str, float] = {}

        #     tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

        #     # reset tr_loss to zero
        #     tr_loss -= tr_loss

        #     logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
        #     logs["learning_rate"] = self._get_learning_rate()

        #     self._total_loss_scalar += tr_loss_scalar
        #     self._globalstep_last_logged = self.state.global_step
        #     self.store_flos()

        #     self.log(logs)

        # Evaluate metrics
        eval_metrics = None
        if True:
            logger.info(f"ignore_keys_for_eval = {ignore_keys_for_eval}")
            eval_metrics = self.evaluate(ignore_keys_for_eval=ignore_keys_for_eval)
            # self._report_to_hp_search(trial, epoch, eval_metrics)
            logger.info(f"cur: {eval_metrics}")

            # Update best metrics
            if eval_metrics[f"eval_{self.test_key}"] > self.best_metrics[f"best_eval_{self.test_key}"]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics[f"best_eval_{self.test_key}"] = eval_metrics[f"eval_{self.test_key}"]

                # Save the best test metrics
                # if self.predict_dataset is not None:
                #     if isinstance(self.predict_dataset, dict):
                #         for dataset_name, dataset in self.predict_dataset.items():
                #             _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                #             self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics[f"test_{self.test_key}"]
                #     else:
                #         _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                #         self.best_metrics[f"best_test_{self.test_key}"] = test_metrics[f"test_{self.test_key}"]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        # Save model checkpoint
        # if self.control.should_save:
        #     self._save_checkpoint(model, trial, metrics=eval_metrics)
        #     self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    
    def log(self, logs: Dict[str, float]):
        """Log a dictionary of metrics."""
        for key, value in logs.items():
            logger.info(f"{key}: {value}")
    
    def train(self, num_epochs=3, resume_from_checkpoint=None, last_checkpoint=None):
        if resume_from_checkpoint:
            checkpoint = resume_from_checkpoint
            self.model.load_state_dict(torch.load(checkpoint))
        num_epochs = int(self.args.num_train_epochs)
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,  # Use the provided data_collator for the train dataset
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            # pin_memory=self.args.dataloader_pin_memory,
        )

        self.model.to(self.device)

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = None
                if "token_type_ids" in batch: 
                    # print("token_type_ids")
                    token_type_ids = batch["token_type_ids"].to(self.device)
                    
                # print(self.model)
                # print(self.model.parameters())
                # pdb.set_trace()
                outputs = self.model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels, 
                    token_type_ids=token_type_ids, 
                    # idx=idx
                )
                print(outputs, file = open('tmpoutput', 'w'))
                loss = outputs.loss
                if torch.__version__ != '2.0.0':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    self.optimizer.step(loss)

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Avg Training Loss: {avg_train_loss:.4f}")

            # Call _maybe_log_save_evaluate at the end of each epoch
            self._maybe_log_save_evaluate(
                tr_loss=running_loss,
                model=self.model,
                trial=None,  # If using hyperparameter search, pass the trial here
                epoch=epoch,
                ignore_keys_for_eval=None
            )

        return self.best_metrics