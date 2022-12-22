import torch
from torch.utils.data import Dataset


class TweetDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tweets = dataframe['text'].values
        self.label = dataframe['label'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    @property
    def n_labels(self):
        return len(set(self.label))

    def __getitem__(self, index):
        tweet = self.tweets[index]
        input = self.tokenizer.__call__(tweet, None, add_special_tokens=True, padding='max_length', 
                                        max_length=self.max_len, truncation=True)
        # convert token ids to tensors
        return {
            'input_ids': torch.tensor(input['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(input['attention_mask'], dtype=torch.long),
            'target': torch.tensor(self.label[index], dtype=torch.long)
        }
