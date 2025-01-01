from prepare_dataset import split_dataset

import os
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset, DatasetDict
from evaluate import load
from sacrebleu.metrics import TER
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader


os.environ["WANDB_DISABLED"] = "true"

model_path = "fine_tuned_model"


def evaluation_with_sacrebleu(model, tokenizer, test_dataloader):
    metric = load("sacrebleu")
    model.eval()

    for i, batch in enumerate(test_dataloader):
        input_ids = torch.stack(batch["input_ids"]).to(model.device)
        outputs = model.generate(input_ids, max_length=160)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references = tokenizer.batch_decode(
            batch["labels"], skip_special_tokens=True)

        metric.add_batch(predictions=predictions, references=[
                         [ref] for ref in references])
        print(f"Batch {i+1}/{len(test_dataloader)} processed")

    print("Evaluation complete. Results:")
    print(metric.compute())


def evaluate_with_rouge(model, tokenizer, test_dataloader):
    rouge = load("rouge")

    model.eval()
    for i, batch in enumerate(test_dataloader):
        input_ids = torch.stack(batch["input_ids"]).to(model.device)
        outputs = model.generate(input_ids, max_length=160)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references = tokenizer.batch_decode(
            batch["labels"], skip_special_tokens=True)
        rouge.add_batch(predictions=predictions, references=references)
        print(f"Batch {i+1}/{len(test_dataloader)} processed")

    print("Evaluation complete. Results:")
    print(rouge.compute())


def evaluate_with_meteor(model, tokenizer, test_dataloader):
    meteor = load("meteor")
    model.eval()
    for i, batch in enumerate(test_dataloader):
        input_ids = torch.stack(batch["input_ids"]).to(model.device)
        outputs = model.generate(input_ids, max_length=160)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references = tokenizer.batch_decode(
            batch["labels"], skip_special_tokens=True)
        meteor.add_batch(predictions=predictions, references=references)
        print(f"Batch {i+1}/{len(test_dataloader)} processed")

    print("Evaluation complete. Results:")
    print(meteor.compute())


def evaluate_with_bleu(model, tokenizer, test_dataloader):
    references = []
    predictions = []

    model.eval()
    for i, batch in enumerate(test_dataloader):
        input_ids = torch.stack(batch["input_ids"]).to(model.device)
        outputs = model.generate(input_ids, max_length=160)
        predictions.extend(tokenizer.batch_decode(
            outputs, skip_special_tokens=True))
        references.extend([[ref] for ref in tokenizer.batch_decode(
            batch["labels"], skip_special_tokens=True)])
        print(f"Batch {i+1}/{len(test_dataloader)} processed")

    print("Evaluation complete. Results:")
    bleu_score = corpus_bleu(references, predictions)
    print(f"BLEU score: {bleu_score}")


def evaluate_with_ter(model, tokenizer, test_dataloader):
    metric = TER()

    references = []
    predictions = []

    model.eval()
    for i, batch in enumerate(test_dataloader):
        input_ids = torch.stack(batch["input_ids"]).to(model.device)
        outputs = model.generate(input_ids, max_length=160)
        predictions.extend(tokenizer.batch_decode(
            outputs, skip_special_tokens=True))
        references.extend([[ref] for ref in tokenizer.batch_decode(
            batch["labels"], skip_special_tokens=True)])
        print(f"Batch {i+1}/{len(test_dataloader)} processed")

    print("Evaluation complete. Results:")
    score = metric.corpus_score(predictions, references)
    print(f"TER score: {score.score}")


def evaluate_with_chrf(model, tokenizer, test_dataloader):
    chrf = load("chrf")

    model.eval()
    for i, batch in enumerate(test_dataloader):
        input_ids = torch.stack(batch["input_ids"]).to(model.device)
        outputs = model.generate(input_ids, max_length=160)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references = tokenizer.batch_decode(
            batch["labels"], skip_special_tokens=True)
        chrf.add_batch(predictions=predictions, references=references)
        print(f"Batch {i+1}/{len(test_dataloader)} processed")

    print("Evaluation complete. Results:")
    print(chrf.compute())


def filter_invalid_tokens(batch, vocab_size, unk_token_id):
    batch["input_ids"] = [
        [token if token < vocab_size else unk_token_id for token in sample]
        for sample in batch["input_ids"]
    ]
    return batch


def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples["source"], max_length=160, truncation=True, padding="max_length")
    labels = tokenizer(
        examples["target"], max_length=160, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def evaluation_metric(mode=1):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_data = pd.read_csv("test_data.csv")
    test_dataset = Dataset.from_pandas(test_data)

    tokenized_test_data = test_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer), batched=True)
    test_dataloader = DataLoader(tokenized_test_data, batch_size=8)

    match mode:
        case 1:
            evaluation_with_sacrebleu(model, tokenizer, test_dataloader)
        case 2:
            evaluate_with_rouge(model, tokenizer, test_dataloader)
        case 3:
            evaluate_with_meteor(model, tokenizer, test_dataloader)
        case 4:
            evaluate_with_bleu(model, tokenizer, test_dataloader)
        case 5:
            evaluate_with_ter(model, tokenizer, test_dataloader)
        case 6:
            evaluate_with_chrf(model, tokenizer, test_dataloader)
        case _:
            evaluation_with_sacrebleu(model, tokenizer, test_dataloader)


def fine_tune_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "minhtoan/t5-translate-vietnamese-nom")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "minhtoan/t5-translate-vietnamese-nom")

    train_dataset, val_dataset, test_dataset = split_dataset()

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    tokenized_data = dataset_dict.map(
        lambda examples: preprocess_function(examples, tokenizer), batched=True)
    tokenizer.add_special_tokens({'unk_token': '<unk>'})
    vocab_size = tokenizer.vocab_size
    unk_token_id = tokenizer.unk_token_id
    tokenized_data = tokenized_data.map(
        lambda batch: filter_invalid_tokens(batch, vocab_size, unk_token_id),
        batched=True
    )

    model.resize_token_embeddings(len(tokenizer))

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")


def main():
    fine_tune_model()
    mode = 3
    evaluation_metric(mode)


if __name__ == "__main__":
    main()
