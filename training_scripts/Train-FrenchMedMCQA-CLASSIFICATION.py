#!/usr/bin/env python3
"""Recipe for training an classification system from textual data using FrenchMedMCQA.
The system had 5 classes (1, 2, 3, 4, 5)
To run this recipe, do the following:
> Change the path of the HuggingFace dataset to your local or remote location.
> Run the training script:
    > python TrainFrenchMedMCQA-CLASSIFICATION-Full.py --model_name="camembert-base"
Authors
 * Yanis LABRAK 2023
"""

import os
import uuid
import argparse

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, roc_auc_score, accuracy_score, classification_report

from datasets import load_dataset, load_metric

import transformers
from transformers import AutoTokenizer, EvalPrediction, AutoModelForSequenceClassification, TrainingArguments, Trainer

print(transformers.__version__)

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-m", "--model_name", help = "HuggingFace Hub model name")
args = vars(parser.parse_args())

dataset_base  = load_dataset("/users/ylabrak/DEFT-2023/Data/Huggingface/FrenchMedMCQA/frenchmedmcqa_full.py")

dataset_train = dataset_base["train"]
print(len(dataset_train))

dataset_val = dataset_base["validation"]
print(len(dataset_val))

dataset_test = dataset_base["test"]
print(len(dataset_test))

metric = load_metric("accuracy")

labels_list = dataset_train.features["number_correct_answers"].names
print(labels_list)

num_labels = len(labels_list)
batch_size = 16
EPOCHS = 1
# EPOCHS = 10
model_checkpoint = str(args["model_name"])

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]

def preprocess_function(e):

    CLS = "<s>"
    BOS = "<s>"
    SEP = "</s>"
    EOS = "</s>"

    text = CLS + " " + e["question"] + f" {SEP} " + f" {SEP} ".join([e[f"answer_{letter}"] for letter in ["a","b","c","d","e"]]) + " " + EOS
    
    res = tokenizer(text, truncation=True, max_length=512, padding="max_length")

    res["label"] = e["number_correct_answers"]

    return res

dataset_train = dataset_train.map(preprocess_function, batched=False)
dataset_train.set_format("torch")

dataset_val   = dataset_val.map(preprocess_function, batched=False)
dataset_val.set_format("torch")

dataset_test   = dataset_test.map(preprocess_function, batched=False)
dataset_test.set_format("torch")

output_dir = f"./models/FrenchMedMCQA-Classification-{model_name}-finetuned-{uuid.uuid4().hex}"

args = TrainingArguments(
    output_dir,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model,
    args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

# ------------------ EVALUATION ------------------

predictions, labels, _ = trainer.predict(dataset_test)
predictions = np.argmax(predictions, axis=1)

# labels_test = [d["number_correct_answers"].item() for d in dataset_test]
ids_test = [d["id"].item() for d in dataset_test]

f1_score = classification_report(
    labels,
    predictions,
    digits=4,
    target_names=labels_list,
)

output_txt = []
output_txt.append("```plain")
output_txt.append(f1_score)
output_txt.append("```")
output_txt = "\n".join(output_txt)
print(output_txt)

f_out_submission = open(f"./submission-{uuid.uuid4().hex}.txt","w")
for indentifier, pred_value in zip(ids_test, predictions):
    f_out_submission.write(f"{indentifier};{pred_value}\n")
f_out_submission.close()
