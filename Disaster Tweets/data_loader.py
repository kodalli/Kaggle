import pandas as pd 
import torch
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics


class DataLoader:
    def __init__(self, texts, targets, max_len=256):
        self.texts = texts
        self.targets = targets
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=False
        )
        self.max_length = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        inputs = self.tokenizer.encode_plus(
            text, 
            None,
            add_special_tokens=True,
            max_length = self.max_length,
            padding="max_length"
        )
        
        ids = inputs["input_ids"]
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_length - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
            }


class BERT(torch.nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", return_dict=False)
        self.drop = torch.nn.Dropout(0.4)
        self.out = torch.nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, pooled_out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids) 
        bert_out = self.drop(pooled_out)
        output = self.out(bert_out)
        return output

def loss_fn(output, targets):
    return torch.nn.BCEWithLogitsLoss()(output, targets.view(-1,1))


def train_func(data_loader, model, optimizer, device, scheduler):
    model.to(device)
    model.train()
    
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
       
        # print(ids)

        optimizer.zero_grad()
        output = model(
            ids=ids,
            mask = mask,
            token_type_ids = token_type_ids
        )
        
        loss = loss_fn(output, targets)
        loss.backward()
        
        optimizer.step()
        scheduler.step()

def eval_func(data_loader, model, device):
    model.eval()
    
    fin_targets = []
    fin_output = []
    
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)


            output = model(
                ids=ids,
                mask = mask,
                token_type_ids = token_type_ids
            )
        
            fin_targets.extend(targets.cpu().detach().numpy().to_list())
            fin_output.extend(torch.sigmoid(output).cpu().detach().numpy().to_list())
            
        return fin_output, fin_targets

def run():
    df = pd.read_csv('data/train.csv')
    data = pd.DataFrame({
        'text' : df['text'],
        'label' : df['target']
    })


    encoder = LabelEncoder()
    data['label'] = encoder.fit_transform(data['label'])

    df_train, df_valid = train_test_split(data, test_size = 0.2, random_state=0, stratify=data.label.values)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = DataLoader(
        texts=df_train.text.values,
        targets=df_train.label.values,
        max_len=160
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=8,
        num_workers=4,
    )

    val_dataset = DataLoader(
        texts=df_valid.text.values,
        targets=df_valid.label.values,
        max_len=160
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=4,
        num_workers=1,
    )

    device = torch.device("cuda")
    model = BERT()

    param_optimizer = list(model.named_parameters())

    no_decay = [
        "bias", 
        "LayerNorm,bias",
        "LayerNorm.weight",
        ]

    optimizer_parameters = [
        {'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
                   'weight_decay':0.001},
        {'params': [p for n,p in param_optimizer if any(nd in n for nd in no_decay)],
                   'weight_decay':0.0}
    ]

    num_train_steps = int(len(df_train)/ 8*10)

    optimizers = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizers,
        num_warmup_steps=0,
        num_training_steps=num_train_steps

    )

    best_accuracy = 0
    for epoch in range(5):
        train_func(data_loader=train_data_loader, model=model, optimizer=optimizers, device=device, scheduler=scheduler)
        outputs, targets = eval_func(data_loader=train_data_loader, model=model, device=device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(outputs, targets)
        print(f"Accuracy Score: {accuracy}")

        if accuracy>best_accuracy:
            torch.save(model.state_dict(), "model.bin")
            best_accuracy = accuracy

def explore():
    df = pd.read_csv('data/train.csv')
    data = pd.DataFrame({
        'text' : df['text'],
        'label' : df['target']
    })

    import matplotlib.pyplot as plt

    plt.style.use('ggplot')

    df['length'] = df['text'].apply(lambda x: len(x))
    plt.hist(df['length'])
    plt.show()


if __name__ == "__main__":
    # run()
    explore()
