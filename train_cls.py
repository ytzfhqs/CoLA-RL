import argparse
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import matthews_corrcoef

class TrainClsModel:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        if 'Qwen' in model_path:
            self.model.config.pad_token_id = 151643
    
    def encoder_text(self, examples, text_column: str):
        encoding = self.tokenizer(
            examples[text_column], max_length=510, padding=True, truncation=True
        )
        return encoding

    def load_data(self) -> DatasetDict:
        train_data = pd.read_csv(
            "cola_data/in_domain_train.tsv",
            sep="\t",
            header=None,
            names=["source", "label", "first_label", "text"],
        )
        test_data = pd.read_csv(
            "cola_data/in_domain_dev.tsv",
            sep="\t",
            header=None,
            names=["source", "label", "first_label", "text"],
        )
        train_ds = Dataset.from_pandas(train_data[["text", "label"]])
        test_ds = Dataset.from_pandas(test_data[["text", "label"]])
        total_ds = DatasetDict({'train': train_ds, 'test': test_ds})
        total_ds = total_ds.map(partial(self.encoder_text, text_column='text'), num_proc=8, remove_columns=["text"])
        return total_ds
    
    @staticmethod
    def compute_metrics(eval_pred) -> dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'mcc': matthews_corrcoef(labels, predictions)}
    
    def train_model(self, train_batch_size, gradient_accumulation_steps, eval_batch_size, num_train_epochs):
        total_ds = self.load_data()
        output_dir = self.model_path.name + "_cls"
        # Qwen系列模型使用cosine学习率调度器
        if 'Qwen' in str(self.model_path):
            lr_scheduler_type = "cosine"
        # bert系列模型使用linear学习率调度器
        else:
            lr_scheduler_type = "linear"
        training_args = TrainingArguments(
            output_dir=output_dir,
            lr_scheduler_type=lr_scheduler_type,
            learning_rate=1.0e-5,
            warmup_ratio=0.01,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_train_epochs,
            logging_steps=0.01,
            eval_steps=0.05,
            eval_strategy="steps",
            save_strategy="epoch",
            report_to="tensorboard",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=total_ds["train"],
            eval_dataset=total_ds["test"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/Qwen3-0.6B")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    args = parser.parse_args()
    
    tcm = TrainClsModel(args.model_path)
    tcm.train_model(args.train_batch_size, args.gradient_accumulation_steps, args.eval_batch_size, args.num_train_epochs)
