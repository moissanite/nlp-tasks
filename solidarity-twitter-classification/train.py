import os
import argparse
import re
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

from data import TweetDataset
from xlm_roberta import XlmRoberta
from config_train import TrainConfig


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Directory path of twitter dataset')
    args = parser.parse_args()
    return args


def load_yml_config(cfg_path):
    with open(cfg_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def data_preprocessing(df):
    for idx, row in df.iterrows():
        text = df.at[idx, 'text']

        # Replace '&amp;' with '&'
        text = re.sub(r'&amp;', '&', text)
        # Remove URL
        text = re.sub(r'https?://\S+|www\.\S+', '',text)
        # Remove numbers
        text = ''.join([i for i in text if not i.isdigit()])    
        # Remove html
        text = re.sub(r'<.*?>', '', text)
        # Remove &gt; <, &lt; >
        text = re.sub(r'&gt;', '',text)
        text = re.sub(r'&lt;', '',text)

        df.at[idx, 'text'] = text

        # require by the task
        if df.at[idx, 'label'] == 3:
            df.at[idx, 'label'] = 2
    return df


def plot_losses(train_losses, valid_losses, file_path):
    plt.plot(train_losses, label='train loss')
    plt.plot(valid_losses, label='val loss')
    plt.legend()
    # plt.show()
    plt.savefig(file_path)


def train(train_dataset):
    total_loss = 0
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size)
    model.train()

    # for batch in enumerate(tqdm(train_dataloader, desc='train')):
    for batch in tqdm(train_dataloader, desc='train'):
        model.zero_grad()
        y_preds = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        loss = model.criterion(y_preds, batch['target'].to(device))
        total_loss += loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss.item() / len(train_dataloader)


def evaluate(dev_dataset):
    total_loss = 0
    predictions = []
    target_labels = []
    dev_dataloader = DataLoader(dev_dataset, batch_size=cfg.train.batch_size)
    model.eval()

    for batch in tqdm(dev_dataloader, desc='val'):
        with torch.no_grad():
            output = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            loss = model.criterion(output, batch['target'].to(device))
        
        total_loss += loss
        pred = torch.argmax(output, dim=-1).tolist()
        predictions.extend(pred)
        target_labels.extend(batch['target'].tolist())
    
    return total_loss.item() / len(dev_dataloader), predictions, target_labels


def run_metrics(model_path, dev_dataset, report_path, cm_path):
    predictions = []
    target_labels = []
    dev_dataloader = DataLoader(dev_dataset, batch_size=cfg.train.batch_size)
    model = torch.load(model_path)
    model.eval()
    with torch.no_grad():
        for batch in dev_dataloader:
            output = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            pred = torch.argmax(output, dim=-1).tolist()
            predictions.extend(pred)
            target_labels.extend(batch['target'].tolist())
    
    print('Classification report:')
    print(metrics.classification_report(target_labels, predictions))
    report = metrics.classification_report(target_labels, predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_excel(report_path)

    # plot confusion matrix
    mcm = metrics.confusion_matrix(target_labels, predictions)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=mcm, display_labels=cfg.dataset.labels)
    cm_display.plot()
    # plt.show()
    plt.savefig(cm_path)


if __name__ == '__main__':
    args = set_argparser()
    config = load_yml_config(args.config)
    cfg = TrainConfig(**config)
    
    set_seed(cfg.train.seed)
    os.makedirs(cfg.model.save_dir, exist_ok=True)
    os.makedirs(cfg.result.dir, exist_ok=True)
    best_checkpoint_path = os.path.join(cfg.model.save_dir, cfg.model.save_best)
    last_checkpoint_path = os.path.join(cfg.model.save_dir, cfg.model.save_last)

    raw_data = pd.read_csv(os.path.join(cfg.dataset.data_dir, 'trainset_v3.csv'))
    train_df = data_preprocessing(raw_data)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    twitter_dataset = TweetDataset(train_df, tokenizer, cfg.model.max_len)
    train_data, dev_data = random_split(twitter_dataset, [0.7, 0.3])
    print(f'dataset size: {len(twitter_dataset)}; train size: {len(train_data)}, dev size: {len(dev_data)}')

    model = XlmRoberta(cfg.model.name, twitter_dataset.n_labels)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    train_losses=[]
    valid_losses=[]
    best_valid_loss = 1000
    lrs = [optimizer.param_groups[0]['lr']]
    for epoch in range(cfg.train.epochs):
        print(f'{"-"*30}\nEpoch {epoch+1}\n{"-"*30}')
        avg_train_loss= train(train_data)
        avg_valid_loss, y_preds, y_trues = evaluate(dev_data)

        print(f'Train Loss: {avg_train_loss: .3f} | Val Loss: {avg_valid_loss: .3f} '
                f'| Val Accuracy: {metrics.accuracy_score(y_trues, y_preds): .3f}')
        print(f'Learning rates: {lrs}\n')

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model, best_checkpoint_path)

        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    
    torch.save(model, last_checkpoint_path)
    plot_losses(train_losses, valid_losses, os.path.join(cfg.result.dir, 'loss.jpg'))

    report_path = os.path.join(cfg.result.dir, 'best_classifiction_report.xlsx')
    cm_path = os.path.join(cfg.result.dir, 'best_cp_cm.jpg')
    run_metrics(best_checkpoint_path, dev_data, report_path, cm_path)
    report_path = os.path.join(cfg.result.dir, 'last_classifiction_report.xlsx')
    cm_path = os.path.join(cfg.result.dir, 'last_cp_cm.jpg')
    run_metrics(last_checkpoint_path, dev_data, report_path, cm_path)
