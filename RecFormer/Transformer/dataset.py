from torch.utils.data import Dataset
import torch
import pandas as pd
import random


class RecDataset(Dataset):
    def __init__(self, df, mode, config):
        super().__init__()
        self.mode = mode
        self.mlm_prob = config['mlm_prob']
        self.max_len = config['max_len']
        self.mask_token = config['track_num']   # set track_num = MASK, 0 is PAD

        df = df[['user_id', 'track_id', 'converted_track_id', 'timestamp']].sort_values('timestamp')
        p = df.groupby('user_id', sort=False)['converted_track_id'].agg(list)
        # sentences = p.values.tolist()
        user_tracks = pd.DataFrame(p)

        self.users, self.user_ids = {}, []
        # idx starts from 1
        for idx, row in user_tracks.iterrows():
            self.users[idx] = row['converted_track_id']
            self.user_ids.append(idx)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_histroy = self.users[user_id]

        tokens, labels = [], []
        for history in user_histroy:
            prob = random.random()

            # [TODO]: set prob that will replace with other items
            if prob < self.mlm_prob:
                tokens.append(self.mask_token)
                labels.append(history)
            else:
                tokens.append(history)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = tokens + [0] * mask_len
        labels = labels + [0] * mask_len

        return torch.LongTensor(tokens), torch.LongTensor(labels)