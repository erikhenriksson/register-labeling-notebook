from transformers import TrainingArguments

training_args = TrainingArguments("test_trainer"),


import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.evaluate()


print("Evaluating with test set...")
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

print(classification_report(trues, preds, target_names=labels))
