#!/usr/bin/env python3
"""Recipe for training an MCQA system from textual data using FrenchMedMCQA.
The multi-label system had 5 classes (1, 2, 3, 4, 5) to choose from
To run this recipe, do the following:
> Change the path of the HuggingFace dataset to your local or remote location.
> Run the training script:
    > python TrainFrenchMedMCQA-QA.py --model_name="camembert-base"
Authors
 * Yanis LABRAK 2023
"""

import os
import uuid
import argparse

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, roc_auc_score, accuracy_score

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

labels_list = dataset_train.features["correct_answers"].feature.names
print(labels_list)

num_labels = len(labels_list)
batch_size = 32
# batch_size = 16
# EPOCHS = 1
EPOCHS = 10
model_checkpoint = str(args["model_name"])

id2label = {idx:label for idx, label in enumerate(labels_list)}
label2id = {label:idx for idx, label in enumerate(labels_list)}

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    problem_type="multi_label_classification",
    num_labels=num_labels,
)

model_name = model_checkpoint.split("/")[-1]

print("#"*50)
print(model.config.max_position_embeddings)
print("#"*50)

def preprocess_function(e):

    CLS = "<s>"
    BOS = "<s>"
    SEP = "</s>"
    EOS = "</s>"

    text = CLS + " " + e["question"] + f" {SEP} " + f" {SEP} ".join([e[f"answer_{letter}"] for letter in ["a","b","c","d","e"]]) + " " + EOS
    
    res = tokenizer(text, truncation=True, max_length=512, padding="max_length")

    labels = [0.0] * 5
    
    for answer_id in e["correct_answers"]:
        labels[answer_id] = 1.0

    res["labels"] = labels

    return res

dataset_train = dataset_train.map(preprocess_function, batched=False)
dataset_train.set_format("torch")

dataset_val   = dataset_val.map(preprocess_function, batched=False)
dataset_val.set_format("torch")

dataset_test   = dataset_test.map(preprocess_function, batched=False)
dataset_test.set_format("torch")

output_dir = f"./models/FrenchMedMCQA-{model_name}-finetuned-{uuid.uuid4().hex}"

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
    push_to_hub=False,
    greater_is_better=True,

    metric_for_best_model="emr",
    # metric_for_best_model="accuracy",
)

def toLogits(predictions, threshold=0.5):

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    return y_pred

def compute_accuracy_exact_match(refs, preds):

    exact_score = []

    for p, r in zip(preds.tolist(), refs.tolist()):
        exact_score.append(p == r)

    return sum(exact_score) / len(exact_score)

def compute_accuracy_hamming(refs, preds):

    scores = []

    for pred, ref in zip(preds.tolist(), refs.tolist()):

        labels_pred = [id2label[i] for i, p in enumerate(pred) if p == 1]
        labels_ref  = [id2label[i] for i, r in enumerate(ref)  if r == 1]

        corrects = sum([p == r for p, r in zip(pred, ref)])
        total_r = len(list(set(labels_pred + labels_ref)))

        scores.append(corrects / total_r)
    
    return sum(scores) / len(scores)

def multi_label_metrics(predictions, labels, threshold=0.5):

    y_pred = toLogits(predictions, threshold)

    y_true = labels

    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    # roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)

    emr = compute_accuracy_exact_match(y_true, y_pred)
    hamming = compute_accuracy_hamming(y_true, y_pred)

    metrics = {'f1': f1_macro_average, 'accuracy': accuracy, 'hamming': hamming, 'emr': emr }
    # metrics = {'f1': f1_macro_average, 'roc_auc': roc_auc, 'accuracy': accuracy, 'hamming': hamming, 'emr': emr }

    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids
    )
    return result

trainer = Trainer(
    model,
    args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

best_model_path = f"{output_dir}/best_model/"
trainer.save_model(best_model_path)
print(f"Best model is saved at : {best_model_path}")

print(trainer.evaluate())

# ------------------ EVALUATION ------------------

predictions, labels, _ = trainer.predict(dataset_test)

print("predictions logits")
predictions = toLogits(predictions, 0.5)
print(predictions)

def toLetters(predictions):
        
    predictions_classes = []

    for p in predictions:
        predictions_classes.append(
            [id2label[i] for i, p in enumerate(p) if p == 1]
        )
    
    return predictions_classes

predictions = toLetters(predictions)

ids_test = [d["id"].item() for d in dataset_test]

f_out_submission = open(f"./submission-MCQA-{uuid.uuid4().hex}.txt","w")
for indentifier, pred_value in zip(ids_test, predictions):
    pred_value = "|".join(pred_value)
    f_out_submission.write(f"{indentifier};{pred_value}\n")
f_out_submission.close()
