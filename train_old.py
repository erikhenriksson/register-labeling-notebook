import transformers
import datasets
import torch
import logging
import sys
import os
import numpy as np
import wandb


logging.disable(logging.INFO)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from pprint import PrettyPrinter
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

pprint = PrettyPrinter(compact=True).pprint

LEARNING_RATE = 0.000022
BATCH_SIZE = 6
TRAIN_EPOCHS = 15
MODEL_NAME = "xlm-roberta-base"
PATIENCE = 5
WORKING_DIR = "/scratch/project_2005092/erik/rl_old"

labels_full = [
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
labels_upper = ["HI", "ID", "IN", "IP", "LY", "MT", "NA", "OP", "SP"]


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model_name", default=MODEL_NAME, help="Pretrained model name")
    ap.add_argument("--train", required=True, help="Path to training data")
    ap.add_argument("--test", required=True, help="Path to test data")
    ap.add_argument(
        "--batch_size",
        metavar="INT",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training",
    )
    ap.add_argument(
        "--epochs",
        metavar="INT",
        type=int,
        default=TRAIN_EPOCHS,
        help="Number of training epochs",
    )
    ap.add_argument(
        "--learning_rate",
        metavar="FLOAT",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate",
    )
    ap.add_argument(
        "--patience",
        metavar="INT",
        type=int,
        default=PATIENCE,
        help="Early stopping patience",
    )
    ap.add_argument("--save_model", default=True, type=bool, help="Save model to file")
    ap.add_argument(
        "--threshold",
        default=None,
        metavar="FLOAT",
        type=float,
        help="threshold for calculating f-score",
    )
    ap.add_argument("--labels", choices=["full", "upper"], default="full")
    ap.add_argument(
        "--load_model", default=None, metavar="FILE", help="load existing model"
    )
    ap.add_argument("--class_weights", default=False, type=bool)
    ap.add_argument("--working_dir", default=WORKING_DIR, help="Working directory")
    ap.add_argument("--tune", default=False, type=bool, help="Tune hyperparameters")

    return ap


options = argparser().parse_args(sys.argv[1:])

if options.labels == "full":
    labels = labels_full
else:
    labels = labels_upper

working_dir = f"{options.working_dir}/{options.train}_{options.test}"

num_labels = len(labels)
print(f"Number of labels: {num_labels}")

# register scheme mapping:
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


def remove_NA(d):
    """Remove null values and separate multilabel values with comma"""
    if d["label"] == None:
        d["label"] = "NA"
    if " " in d["label"]:
        d["label"] = ",".join(sorted(d["label"].split()))
    return d


def label_encoding(d):
    """Split the multi-labels"""
    d["label"] = d["label"].split(",")
    mapped = [sub_register_map[l] if l not in labels else l for l in d["label"]]
    d["label"] = np.array(sorted(list(set(mapped))))
    return d


def binarize(dataset):
    """Binarize the labels of the data. Fitting based on the whole data."""
    mlb = MultiLabelBinarizer()
    mlb.fit([labels])
    print("Binarizing the labels")
    dataset = dataset.map(lambda line: {"label": mlb.transform([line["label"]])[0]})
    return dataset


data_files = {"train": [], "dev": [], "test": []}

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

# data column structures
cols = {
    "fr": ["a", "b", "label", "text", "c"],
    "fi": ["label", "text", "a", "b", "c"],
    "sv": ["a", "b", "label", "text", "c"],
}


# choose data with all languages with option 'multi'
for l in options.train.split("-"):
    print("L train", l)
    data_files["train"].append(f"data/{l}/train.tsv")
    if not (l == "multi" or l in small_languages):
        print("dev", l)
        data_files["dev"].append(f"data/{l}/dev.tsv")
for l in options.test.split("-"):
    print("L test", l)
    # check if zero-shot for small languages, if yes then test with full data
    if l in small_languages and not (
        l in options.train.split("-") or "multi" in options.train.split("-")
    ):
        data_files["test"].append(f"data/{l}/{l}.tsv")
    else:
        data_files["test"].append(f"data/{l}/test.tsv")
print("datafiles", data_files)
dataset = datasets.load_dataset(
    "csv",
    data_files=data_files,  # {'train':options.train, 'test':options.test, 'dev': options.dev},
    delimiter="\t",
    column_names=cols[options.train],
    features=datasets.Features(
        {  # Here we tell how to interpret the attributes
            "text": datasets.Value("string"),
            #      "label":datasets.Value("int32")
            "label": datasets.Value("string"),
        }
    ),
    cache_dir=f"{working_dir}/dataset_cache",
)
dataset = dataset.shuffle(seed=42)

# smaller tests
# dataset["train"]=dataset["train"].select(range(400))
# dataset["dev"]=dataset["dev"].select(range(100))
# pprint(dataset['test']['label'][:10])
dataset = dataset.map(remove_NA)
# remove examples that have more than four labels
# dataset = dataset.filter(lambda example: len(example['label'].split(','))<=4) #WE WANNA HAVE THOSE
# dataset = dataset.filter(lambda example: 'MT' not in example['label'].split(',') and 'OS' not in example['label'].split(',')) # WE WANNA HAVE THOSE TOO
print("XXX BEFORE")
print(dataset["train"][0])
dataset = dataset.map(label_encoding)
print("XXX AFTER")
print(dataset["train"][0])


def compute_class_weights(dataset):
    freqs = [0] * len(labels)
    print("FREQS", freqs)
    n_examples = len(dataset["train"])
    print("LEN train dataset", len(dataset["train"]))
    print("LABELS", labels)
    print("LEN LABELS", len(labels))
    for e in dataset["train"]["label"]:
        #       print("EE", e)
        for i in range(len(labels)):
            #            print("E i", e[i])
            if e[i] != 0:
                freqs[i] += 1
    weights = []
    # print("FREQS 2", freqs)
    for i in range(len(labels)):  # , label in enumerate(labels):
        #        print("III", i)
        print(freqs[i])
        try:
            weights.append(n_examples / (len(labels) * freqs[i]))
        except:
            weights.append(0.0)
    print("weights:", weights)
    class_weights = torch.FloatTensor(weights).cuda()
    return class_weights


# class_weights = compute_class_weights(dataset)

dataset = binarize(dataset)
print("XXX labels")
print("XXX sample")
# print(dataset)
print(dataset["train"][0])
print(dataset["train"][1])
print("text")
print(dataset["train"]["text"][0])
print("dataset")
print(dataset)
# pprint(dataset['test']['label'][:5])
# pprint(dataset['test']['text'][:5])
if options.class_weights is True:
    class_weights = compute_class_weights(dataset)

# config=AutoConfig.from_pretrained("XXX", output_hidden_states=True)
# config = config_class.from_pretrained(options.model_name)
# config.output_hidden_states = True
# model = model.from_pretrained(name, config=config)
model_name = options.model_name  # "xlm-roberta-base"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
print("tokenizer and model ok")


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding=True,
        #        return_tensors='pt'
    )


# Apply the tokenizer to the whole dataset using .map()
# dataset = dataset.map(tokenize)
# print(dataset['test'][0])

# evaluate only
if options.load_model is not None:
    model = torch.load(options.load_model)
    #    torch.device
    model.to("cpu")
    trues = dataset["test"]["label"]
    inputs = dataset["test"]["text"]
    pred_labels = []
    for index, i in enumerate(inputs):
        tok = tokenizer(i, truncation=True, max_length=512, return_tensors="pt")
        pred = model(**tok)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(pred.logits.detach().numpy()))
        preds = np.zeros(probs.shape)
        preds[np.where(probs >= options.threshold)] = 1
        pred_labels.extend(preds)
    #        print("preds",[labels[idx] for idx, label in enumerate(preds.flatten()) if label >= options.threshold])
    #        print("trues",[labels[idx] for idx, label in enumerate(trues[index]) if label >= options.threshold])
    #        print(i)
    #    print(pred_labels)
    #    print(trues)
    print("F1-score", f1_score(y_true=trues, y_pred=pred_labels, average="micro"))
    print(classification_report(trues, pred_labels, target_names=labels))
    sys.exit()
    # return [labels[idx] for idx, label in enumerate(preds) if label >= options.threshold]

# Apply the tokenizer to the whole dataset using .map()
dataset = dataset.map(tokenize)
print("dataset tokenized")

# set up a separated directory for caching
# config=AutoConfig.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1", output_hidden_states=True)
# config = config.output_hidden_states=True
# config = config_class.from_pretrained(model_name)
# config.output_hidden_states = True
# model = model.from_pretrained(name, config=config)


def model_init():
    return transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=f"{working_dir}/model_cache",
    )  # , config=config)#config.output_hidden_states=True)
    # assert model.config.output_hidden_states == True


class MultilabelTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if options.class_weights == True:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.float().view(-1, self.model.config.num_labels),
        )
        return (loss, outputs) if return_outputs else loss


print("Model type: ", options.model_name)
print("Learning rate: ", options.learning_rate)
print("Batch size: ", options.batch_size)
print("Epochs: ", options.epochs)

trainer_args = transformers.TrainingArguments(
    f"{working_dir}/checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    eval_steps=100,
    logging_steps=100,
    learning_rate=options.learning_rate,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    per_device_train_batch_size=options.batch_size,
    per_device_eval_batch_size=32,
    num_train_epochs=options.epochs,
    report_to="wandb" if options.tune else None,
)

data_collator = transformers.DataCollatorWithPadding(tokenizer)

# Argument gives the number of steps of patience before early stopping
early_stopping = transformers.EarlyStoppingCallback(early_stopping_patience=5)

threshold = options.threshold


# in case a threshold was not given, choose the one that works best with the evaluated data
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


def multi_label_metrics(predictions, labels, threshold):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_th05 = np.zeros(probs.shape)
    y_th05[np.where(probs >= 0.5)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {
        "f1": f1_micro_average,  # user-chosen or optimized threshold
        "f1_th05": f1_score(
            y_true=y_true, y_pred=y_th05, average="micro"
        ),  # report also f1-score with threshold 0.5
        "roc_auc": roc_auc,
        "accuracy": accuracy,
    }
    return metrics


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    if options.threshold == None:
        best_f1_th = optimize_threshold(preds, p.label_ids)
        threshold = best_f1_th
        print("Best threshold:", threshold)
    result = multi_label_metrics(
        predictions=preds, labels=p.label_ids, threshold=threshold
    )
    return result


class LogSavingCallback(transformers.TrainerCallback):
    def on_train_begin(self, *args, **kwargs):
        self.logs = defaultdict(list)
        self.training = True

    def on_train_end(self, *args, **kwargs):
        self.training = False

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if self.training:
            for k, v in logs.items():
                if k != "epoch" or v not in self.logs[k]:
                    self.logs[k].append(v)


training_logs = LogSavingCallback()
threshold = options.threshold


def get_trainer():
    return MultilabelTrainer(
        model=None,
        model_init=model_init,
        args=trainer_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[early_stopping, training_logs],
    )


if options.tune:
    from ray import tune
    from ray.tune.schedulers import PopulationBasedTraining

    # from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune import CLIReporter

    """
    asha_scheduler = tune.schedulers.ASHAScheduler(
        metric="eval_f1",
        mode="max",
    )

    hyperopt_search = HyperOptSearch(metric="eval_f1", mode="max")

    tune_config = {
        # "learning_rate": tune.grid_search(
        #    [0.00008, 0.00006, 0.00004, 0.00002, 0.000008]
        # ),
        "learning_rate": tune.loguniform(upper=0.0001, lower=1e-07),
        # "weight_decay": tune.choice([0.0, 0.1, 0.2, 0.3]),
        # "num_train_epochs": tune.choice([20]),
        "per_device_train_batch_size": tune.choice([6, 8, 10, 12]),
    }

    trainer = get_trainer()

    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        scheduler=asha_scheduler,
        search_alg=hyperopt_search,
        direction="maximize",
    )

    """

    tune_config = {
        "per_device_eval_batch_size": 32,
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_f1",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "learning_rate": [0.0001, 0.000067, 0.000033, 0.00001, 0.0000067],
            "per_device_train_batch_size": [6, 8, 10],
        },
    )

    reporter = CLIReporter(
        parameter_columns={
            "learning_rate": "lr",
            "per_device_train_batch_size": "bs",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=[
            "eval_f1",
            "eval_f1_th05",
            "eval_accuracy",
            "eval_loss",
            "epoch",
            "training_iteration",
        ],
    )

    trainer = get_trainer()

    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        scheduler=scheduler,
        keep_checkpoints_num=1,
        local_dir=f"{working_dir}/ray",
        name="tune_transformer_pbt",
        log_to_file=True
        # checkpoint_config=CheckpointConfig(
        #    num_to_keep=1,
        #    checkpoint_score_attribute="training_iteration",
        # ),
        # progress_reporter=reporter,
        # local_dir="~/ray_results/",
        # name="tune_transformer_pbt",
        # log_to_file=True,
    )

    """
    wandb.login()
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = "register-labeling"

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "true"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    params = {
        "method": "bayes",
        "metric": {"name": "eval_f1", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-7,
                "max": 1e-4,
            },
            "per_device_train_batch_size": {"values": [6, 8, 12, 16]},
        },
    }

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            trainer = get_trainer()
            trainer.train()

    sweep_id = wandb.sweep(params, project="register-labeling")

    wandb.agent(sweep_id, train, count=20)

    exit()
    """


print("Training...")
trainer = get_trainer()
trainer.train()

print("Evaluating with test set...")
eval_results = trainer.evaluate(dataset["test"])

pprint(eval_results)

test_pred = trainer.predict(dataset["test"])
trues = test_pred.label_ids
predictions = test_pred.predictions
# print("true:")
# print(trues)
if threshold == None:
    threshold = optimize_threshold(predictions, trues)
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions))
# next, use threshold to turn them into integer predictions
preds = np.zeros(probs.shape)
preds[np.where(probs >= threshold)] = 1

# if you want to check the predictions
# for i, (t, p) in enumerate(zip(trues,preds)):
#  print("true", [labels[idx] for idx, label in enumerate(t) if label == 1])
#  print("pred", [labels[idx] for idx, label in enumerate(p) if label > threshold])
#  print(dataset['test']['text'][i])


print(classification_report(trues, preds, target_names=labels))

if options.save_model:
    torch.save(
        trainer.model,
        f"{working_dir}/saved_model",
    )
