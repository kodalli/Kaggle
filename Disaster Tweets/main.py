import torch
import torch.nn as nn
import transformers
import tez
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd


class BERTDataset:
    def __init__(self, texts, targets, max_len=256):
        self.texts = texts
        self.targets = targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=False
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )
        resp = {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float)
        }
        return resp


class TextModel(tez.Model):
    def __init__(self, num_classes, num_train_steps):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(
            "bert-base-uncased",
            return_dict=False
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
        self.num_train_steps = num_train_steps

    def fetch_optimizer(self):
        opt = AdamW(self.parameters(), lr=1e-4)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def monitor_metrics(self, outputs, targets):
        outputs = torch.sigmoid(outputs).cpu().detach().numpu() >= 0.5
        targets = targets.cpu().detach().numpu()
        return {
            "accuracy": metrics.accuracy_score(targets, outputs)
        }

    def forward(self, ids, mask, token_type_ids, targets=None):
        _, x = self.bert(ids, attention_mask=mask,
                         token_type_ids=token_type_ids)
        x = self.bert_drop(x)
        x = self.out(x)
        if targets is not None:
            loss = self.loss(x, targets)
            met = self.monitor_metrics(x, targets)
            return x, loss, met
        return x, 0, {}


def train_model(fold):
    df = pd.read_csv("data/train.csv")
    # df_train = df[df.kfold!=0].reset_index(drop=True)
    # df_valid = df[df.kfold!=0].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['target'], test_size=0.2)

    train_dataset = BERTDataset(X_train, y_train)
    valid_dataset = BERTDataset(X_test, y_test)

    TRAIN_BS = 32
    EPOCHS = 10

    n_train_steps = int(len(X_train) / TRAIN_BS * EPOCHS)
    model = TextModel(num_classes=1, num_train_steps=n_train_steps)

    es = tez.callbacks.EarlyStopping(
        monitor="valid_loss", patience=3, model_path="model.bin")
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        device="cpu",
        epochs=EPOCHS,
        train_bs=TRAIN_BS,
        callbacks=[es]
    )
    # model.load("model.bin", device="cpu")
    preds = model.predict(valid_dataset)


if __name__ == "__main__":
    train_model(fold=0)
