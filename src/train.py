import os
import json
import torch
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from evaluate import load as load_metric
import numpy as np


def load_data(train_file, test_file):
    train_data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))

    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))

    return Dataset.from_list(train_data), Dataset.from_list(test_data)


def train_model():

    print("Using device:", "GPU" if torch.cuda.is_available() else "CPU")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')

    tokenizer.add_tokens(['<hi>', '<bn>', '<ta>'])
    model.resize_token_embeddings(len(tokenizer))

    print("Loading data...")
    train_dataset, test_dataset = load_data('data/train.jsonl', 'data/test.jsonl')

    train_dataset = train_dataset.shuffle(seed=42).select(range(10000))
    test_dataset = test_dataset.shuffle(seed=42).select(range(1000))

    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    def preprocess(examples):
        inputs = tokenizer(
            examples['source'],
            max_length=64,
            truncation=True,
            padding=False
        )
        labels = tokenizer(
            examples['target'],
            max_length=64,
            truncation=True,
            padding=False
        )
        inputs['labels'] = labels['input_ids']
        return inputs

    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=['source', 'target']
    )

    test_dataset = test_dataset.map(
        preprocess,
        batched=True,
        remove_columns=['source', 'target']
    )

    # Load metrics once outside compute_metrics to avoid reloading every eval step
    cer_metric = load_metric("cer")

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds

        decoded_preds = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True
        )

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True
        )

        exact_matches = sum(
            p.strip() == l.strip()
            for p, l in zip(decoded_preds, decoded_labels)
        )

        accuracy = exact_matches / len(decoded_preds)
        cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            'accuracy': accuracy,
            'cer': cer
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir='models/transliteration',
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=5e-4,
        warmup_steps=100,
        logging_steps=50,
        evaluation_strategy='steps',
        eval_steps=200,
        save_steps=400,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=64,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True
        ),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Training...")
    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained('models/transliteration')

    results = trainer.evaluate()

    print(f"\nAccuracy: {results['eval_accuracy']:.4f}")
    print(f"CER: {results['eval_cer']:.4f}")

    with open('models/transliteration/results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    train_model()