from torch.utils.data import Dataset
import torch
import pandas as pd
import random


class RecDataset(Dataset):
    def __init__(self, df, user_info, mode, config, test_df=None):
        super().__init__()
        self.mode = mode
        self.mlm_prob = config['mlm_prob']
        self.max_len = config['max_len']
        self.mask_token = config['track_num']   # set track_num = MASK, 0 is PAD
        self.artist_mask_token = config['artist_num']   # set artist_num = MASK, 0 is PAD

        df = df[['user_id', 'converted_track_id', 'converted_artist_id', 'timestamp']].sort_values('timestamp')
        p = df.groupby('user_id', sort=False)[['converted_track_id', 'converted_artist_id']].agg(list)
        # sentences = p.values.tolist()
        user_tracks = pd.DataFrame(p).join(user_info, on="user_id", how='left')

        self.users, self.user_ids = {}, []
        if mode == 'train':
            # idx starts from 1
            for idx, row in user_tracks.iterrows():
                self.users[idx] = (row['converted_track_id'], row['converted_artist_id'], row['converted_gender'], row['converted_country'], row['novelty_artist_avg_month'])
                self.user_ids.append(idx)
        elif mode == 'test':
            # idx starts from 1
            for idx, row in user_tracks.loc[test_df].iterrows():
                self.users[idx] = (row['converted_track_id'], row['converted_artist_id'], row['converted_gender'], row['converted_country'], row['novelty_artist_avg_month'])
                self.user_ids.append(idx)
        else:
            raise NotImplementedError()
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_histroy, user_artist, user_gender, user_country, novelty_artist_avg_month = self.users[user_id]

        if self.mode == 'train':
            tokens, artists, genders, countrys, artist_avg_months, labels = [], [], [], [], [], []
            for idx, (history, artist) in enumerate(zip(user_histroy, user_artist)):
                prob = random.random()

                # # always masked the last token
                # if idx == len(user_histroy) - 1:
                #     tokens.append(self.mask_token)
                #     artists.append(artist)
                #     genders.append(user_gender)
                #     countrys.append(user_country)
                #     labels.append(history)
                #     continue

                # [TODO]: set prob that will replace with other items
                if prob < self.mlm_prob:
                    tokens.append(self.mask_token)
                    artists.append(artist)
                    genders.append(user_gender)
                    countrys.append(user_country)
                    artist_avg_months.append(novelty_artist_avg_month)
                    labels.append(history)
                else:
                    tokens.append(history)
                    artists.append(artist)
                    genders.append(user_gender)
                    countrys.append(user_country)
                    artist_avg_months.append(novelty_artist_avg_month)
                    labels.append(0)

            tokens = tokens[-self.max_len:]
            artists = artists[-self.max_len:]
            genders = genders[-self.max_len:]
            countrys = countrys[-self.max_len:]
            artist_avg_months = artist_avg_months[-self.max_len:]
            labels = labels[-self.max_len:]

            mask_len = self.max_len - len(tokens)

            tokens = [0] * mask_len + tokens
            artists = [0] * mask_len + artists
            genders = [0] * mask_len + genders
            countrys = [0] * mask_len + countrys
            artist_avg_months = [0] * mask_len + artist_avg_months
            labels = [0] * mask_len + labels

            return torch.LongTensor(tokens), torch.LongTensor(artists), torch.LongTensor(genders), torch.LongTensor(countrys), torch.FloatTensor(artist_avg_months), torch.LongTensor(labels)
        elif self.mode == 'test':
            user_histroy += [self.mask_token]
            tokens = user_histroy[-self.max_len:]

            # pad target artist with mask but it won't learn in training (not the best way)
            user_artist += [self.artist_mask_token]
            artists = user_artist[-self.max_len:]

            genders = [user_gender] * len(artists)
            countrys = [user_country] * len(artists)
            artist_avg_months = [novelty_artist_avg_month] * len(artists)

            mask_len = self.max_len - len(tokens)
            tokens = [0] * mask_len + tokens
            artists = [0] * mask_len + artists
            genders = [0] * mask_len + genders
            countrys = [0] * mask_len + countrys
            artist_avg_months = [0] * mask_len + artist_avg_months

            return user_id, torch.LongTensor(tokens), torch.LongTensor(artists), torch.LongTensor(genders), torch.LongTensor(countrys), torch.FloatTensor(artist_avg_months)
        else:
            raise NotImplementedError