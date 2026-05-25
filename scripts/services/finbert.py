import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


def fine_tune_finbert(X_train, y_train):
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    train_df = pd.DataFrame({"text": X_train_final, "label": y_train_final})
    val_df = pd.DataFrame({"text": X_val, "label": y_val})

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    tokenized_train = tokenized_train.remove_columns(["text", "__index_level_0__"])
    tokenized_val = tokenized_val.remove_columns(["text", "__index_level_0__"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"accuracy": acc, "f1": f1}

    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert", num_labels=3, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="models/fin_bert_checkpoint",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    print("-----fine-tuning FinBERT-----")
    trainer.train()

    save_path = "models/fin_bert_finetuned"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

