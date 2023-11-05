import numpy as np
import torch
import datasets
from pprint import PrettyPrinter
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from ray import tune

data_path = "data/"
output_path = "/scratch/project_2005092/erik/register-labeling-notebook/"
evaluations = {
    "xmlr-base-fr-best": {
        "save_model": f"{output_path}fr/model",
        "model_name": "xlm-roberta-base",
        "train": "fr",
        "test": "fr",
        "columns": ["a", "b", "label", "text", "c"],
        "class_weights": False,
        "lr": 3.2708e-05,
        "train_batch_size": 8,
        "eval_batch_size": 32,
        "weight_decay": 0,
        "epochs": 50,
        "patience": 5,
        "threshold": None,
        "cache_dir": f"{output_path}fr/cache",
        "checkpoint_dir": f"{output_path}fr/checkpoints",
        "tune_hyperparameters": False,
    },
    "xmlr-base-fr-tune": {
        "model_name": "xlm-roberta-base",
        "train": "fr",
        "test": "fr",
        "columns": ["a", "b", "label", "text", "c"],
        "class_weights": False,
        "lr": 3.2708e-05,
        "train_batch_size": 8,
        "eval_batch_size": 32,
        "weight_decay": 0,
        "epochs": 50,
        "patience": 5,
        "threshold": None,
        "cache_dir": f"{output_path}fr_tune/cache",
        "checkpoint_dir": f"{output_path}fr_tune/checkpoints",
        "tune_hyperparameters": True,
    },
}

# only train and test for these languages
small_languages = [
    "ar",
    "ca",
    "es",
    "fa",
    "hi",
    "id",
    "jp",
    "no",
    "pt",
    "tr",
    "ur",
    "zh",
]

labels = [
    "HI",
    "ID",
    "IN",
    "IP",
    "LY",
    "MT",
    "NA",
    "OP",
    "SP",
    "av",
    "ds",
    "dtp",
    "ed",
    "en",
    "fi",
    "it",
    "lt",
    "nb",
    "ne",
    "ob",
    "ra",
    "re",
    "rs",
    "rv",
    "sr",
]

sub_register_map = {
    "NA": "NA",
    "NE": "ne",
    "SR": "sr",
    "PB": "nb",
    "HA": "NA",
    "FC": "NA",
    "TB": "nb",
    "CB": "nb",
    "OA": "NA",
    "OP": "OP",
    "OB": "ob",
    "RV": "rv",
    "RS": "rs",
    "AV": "av",
    "IN": "IN",
    "JD": "IN",
    "FA": "fi",
    "DT": "dtp",
    "IB": "IN",
    "DP": "dtp",
    "RA": "ra",
    "LT": "lt",
    "CM": "IN",
    "EN": "en",
    "RP": "IN",
    "ID": "ID",
    "DF": "ID",
    "QA": "ID",
    "HI": "HI",
    "RE": "re",
    "IP": "IP",
    "DS": "ds",
    "EB": "ed",
    "ED": "ed",
    "LY": "LY",
    "PO": "LY",
    "SO": "LY",
    "SP": "SP",
    "IT": "it",
    "FS": "SP",
    "TV": "SP",
    "OS": "OS",
    "IG": "IP",
    "MT": "MT",
    "HT": "HI",
    "FI": "fi",
    "OI": "IN",
    "TR": "IN",
    "AD": "OP",
    "LE": "OP",
    "OO": "OP",
    "MA": "NA",
    "ON": "NA",
    "SS": "NA",
    "OE": "IP",
    "PA": "IP",
    "OF": "ID",
    "RR": "ID",
    "FH": "HI",
    "OH": "HI",
    "TS": "HI",
    "OL": "LY",
    "PR": "LY",
    "SL": "LY",
    "TA": "SP",
    "OTHER": "OS",
    "": "",
}


def get_data(evaluation):
    data_files = {"train": [], "dev": [], "test": []}

    for l in evaluation["train"].split("-"):
        data_files["train"].append(f"{data_path}{l}/train.tsv")
        if not (l in small_languages):
            data_files["dev"].append(f"{data_path}{l}/dev.tsv")
        else:
            # Small languages use test as dev
            data_files["dev"].append(f"{data_path}{l}/test.tsv")

    for l in evaluation["test"].split("-"):
        # check if zero-shot for small languages, if yes then test with full data
        if l in small_languages and not (l in evaluation["train"].split("-")):
            data_files["test"].append(f"{data_path}{l}/{l}.tsv")
        else:
            data_files["test"].append(f"{data_path}{l}/test.tsv")

    return data_files


def compute_class_weights(dataset):
    freqs = [0] * len(labels)
    n_examples = len(dataset["train"])

    for e in dataset["train"]["label"]:
        for i in range(len(labels)):
            if e[i] != 0:
                freqs[i] += 1
    weights = []

    for i in range(len(labels)):
        try:
            weights.append(n_examples / (len(labels) * freqs[i]))
        except:
            weights.append(0.0)
    print("weights:", weights)
    class_weights = torch.FloatTensor(weights)
    return class_weights


def get_class_frequencies(dataset):
    y = [0] * len(labels)

    for example in dataset["train"]:
        for i, val in enumerate(example["label"]):
            y[i] += int(val.item())

    expanded_y = [index for index, count in enumerate(y) for _ in range(count)]

    return expanded_y


def optimize_threshold(predictions, labels):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    best_f1 = 0
    best_f1_threshold = 0.5  # use 0.5 as a default threshold
    y_true = labels
    for th in np.arange(0.3, 0.7, 0.05):
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= th)] = 1
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = th
    return best_f1_threshold


evaluation_name = "xmlr-base-fr"

pprint = PrettyPrinter(compact=True).pprint

# Init data

evaluation = evaluations[evaluation_name]
data_files = get_data(evaluation)

id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

print(f"Data files: {data_files}")

# Prepare dataset

dataset = datasets.load_dataset(
    "csv",
    data_files=data_files,
    delimiter="\t",
    column_names=evaluation["columns"],
    features=datasets.Features(
        {x: datasets.Value("string") for x in evaluation["columns"]}
    ),
    cache_dir=evaluation["cache_dir"],
    on_bad_lines="skip",
)


tokenizer = AutoTokenizer.from_pretrained(evaluation["model_name"])


def preprocess_data(example):
    text = example["text"] or ""
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    mapped_labels = set(
        [
            sub_register_map[l] if l not in labels else l
            for l in (example["label"] or "NA").split()
        ]
    )
    encoding["label"] = np.array([1.0 if l in mapped_labels else 0.0 for l in labels])
    return encoding


dataset = dataset.shuffle(seed=42)
dataset = dataset.map(preprocess_data, remove_columns=["a", "b", "c"])
dataset.set_format("torch")

# Train


if evaluation["class_weights"] == True:
    from sklearn.utils.class_weight import compute_class_weight

    print(list(range(0, len(labels))))
    print(get_class_frequencies(dataset))
    class_weights = compute_class_weight(
        "balanced",
        classes=list(range(0, len(labels))),
        y=get_class_frequencies(dataset),
    )

    class_weights = torch.FloatTensor(class_weights)


class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        if evaluation["class_weights"] == True:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
        evaluation["model_name"],
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )


training_args = TrainingArguments(
    evaluation["checkpoint_dir"],
    learning_rate=evaluation["lr"],
    per_device_train_batch_size=evaluation["train_batch_size"],
    per_device_eval_batch_size=evaluation["eval_batch_size"],
    num_train_epochs=evaluation["epochs"],
    weight_decay=evaluation["weight_decay"],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    eval_steps=100,
)


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1  # configured threshold
    y_pred_th05 = np.zeros(probs.shape)
    y_pred_th05[np.where(probs >= 0.5)] = 1  # default threshold
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    f1_micro_average_th05 = f1_score(y_true=y_true, y_pred=y_pred_th05, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {
        "f1": f1_micro_average,
        "f1_th0.5": f1_micro_average_th05,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "threshold": threshold,
    }
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    threshold = (
        evaluation["threshold"]
        if evaluation["threshold"]
        else optimize_threshold(preds, p.label_ids)
    )

    result = multi_label_metrics(
        predictions=preds, labels=p.label_ids, threshold=threshold
    )
    return result


# Argument gives the number of steps of patience before early stopping
early_stopping = EarlyStoppingCallback(early_stopping_patience=evaluation["patience"])

trainer = MultiLabelTrainer(
    model=None,
    model_init=model_init,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

# Tune hyperparameters or just train

if evaluation["tune_hyperparameters"]:
    asha_scheduler = tune.schedulers.ASHAScheduler(
        metric="eval_f1",
        mode="max",
    )

    tune_config = {
        "learning_rate": tune.grid_search(
            [0.0001, 0.00008, 0.00006, 0.00004, 0.00002, 0.000001]
        ),
        # "weight_decay": tune.choice([0.0, 0.1, 0.2, 0.3]),
        # "num_train_epochs": tune.choice([20]),
        "per_device_train_batch_size": tune.choice([6, 8, 10, 12, 14]),
    }

    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config, backend="ray", scheduler=asha_scheduler
    )

else:
    trainer.train()

    if evaluation["save_model"]:
        trainer.save_model(evaluation["save_model"])

print("Evaluating with test set... (last threshold)")
eval_results = trainer.evaluate(dataset["test"])


pprint(eval_results)


test_pred = trainer.predict(dataset["test"])
trues = test_pred.label_ids
predictions = test_pred.predictions

if not evaluation["threshold"]:
    threshold = optimize_threshold(predictions, trues)
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions))

preds = np.zeros(probs.shape)
preds[np.where(probs >= threshold)] = 1

print(f"Evaluating with test set... (optimized threshold {threshold})")

print(classification_report(trues, preds, target_names=labels))
