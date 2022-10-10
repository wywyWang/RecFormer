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

        df = df[['user_id', 'converted_track_id', 'timestamp', 'hour']].sort_values('timestamp')
        p = df.groupby('user_id', sort=False)[['converted_track_id', 'hour']].agg(list)
        p['most_hour'] = p['hour'].apply(lambda x: max(set(x), key=x.count))
        user_tracks = pd.DataFrame(p).join(user_info, on="user_id", how='left')

        self.users, self.user_ids = {}, []
        if mode == 'train':
            # idx starts from 1
            for idx, row in user_tracks.iterrows():
                self.users[idx] = (row['converted_track_id'], row['converted_gender'], row['converted_country'], row['hour'], row['most_hour'], row['age_bin'])
                self.user_ids.append(idx)
        elif mode == 'test':
            # idx starts from 1
            for idx, row in user_tracks.loc[test_df].iterrows():
                self.users[idx] = (row['converted_track_id'], row['converted_gender'], row['converted_country'], row['hour'], row['most_hour'], row['age_bin'])
                self.user_ids.append(idx)
        else:
            raise NotImplementedError()
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_histroy, user_gender, user_country, user_hour, user_most_hour, user_age = self.users[user_id]

        if self.mode == 'train':
            tokens, genders, countrys, hours, ages, labels = [], [], [], [], [], []
            for idx, history in enumerate(user_histroy):
                prob = random.random()

                # # always masked the last token
                # if idx == len(user_histroy) - 1:
                #     tokens.append(self.mask_token)
                #     genders.append(user_gender)
                #     countrys.append(user_country)
                #     labels.append(history)
                #     novelty_artist_avg_months.append(novelty_artist_avg_month)
                #     novelty_artist_avg_6months.append(novelty_artist_avg_6month)
                #     novelty_artist_avg_years.append(novelty_artist_avg_year)
                #     continue

                if prob < self.mlm_prob:
                    tokens.append(self.mask_token)
                    labels.append(history)
                else:
                    tokens.append(history)
                    labels.append(0)
                genders.append(user_gender)
                countrys.append(user_country)
                ages.append(user_age)
                hours.append(user_hour[idx])

            tokens = tokens[-self.max_len:]
            genders = genders[-self.max_len:]
            countrys = countrys[-self.max_len:]
            ages = ages[-self.max_len:]
            hours = hours[-self.max_len:]
            labels = labels[-self.max_len:]

            mask_len = self.max_len - len(tokens)

            tokens = [0] * mask_len + tokens
            genders = [0] * mask_len + genders
            countrys = [0] * mask_len + countrys
            ages = [0] * mask_len + ages
            hours = [0] * mask_len + hours
            labels = [0] * mask_len + labels

            return torch.LongTensor(tokens), torch.LongTensor(genders), torch.LongTensor(countrys), torch.LongTensor(hours), torch.LongTensor(ages), torch.LongTensor(labels)
        elif self.mode == 'test':
            # put at the last timestamp
            user_histroy += [self.mask_token]
            tokens = user_histroy[-self.max_len:]

            genders = [user_gender] * len(tokens)
            countrys = [user_country] * len(tokens)
            ages = [user_age] * len(tokens)

            # assume the user listen in the hour he/her often listen
            user_hour += [user_most_hour]
            hours = user_hour[-self.max_len:]

            mask_len = self.max_len - len(tokens)
            tokens = [0] * mask_len + tokens
            genders = [0] * mask_len + genders
            countrys = [0] * mask_len + countrys
            ages = [0] * mask_len + ages
            hours = [0] * mask_len + hours

            return user_id, torch.LongTensor(tokens), torch.LongTensor(genders), torch.LongTensor(countrys), torch.LongTensor(hours), torch.LongTensor(ages)
        else:
            raise NotImplementedError