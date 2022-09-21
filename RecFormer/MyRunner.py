import numpy as np
import pandas as pd
from reclist.abstractions import RecModel
from .Transformer.transformer import TransformerEncoder
from .Transformer.dataset import RecDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import torch
import torch.nn as nn
import os


class RecRunner(RecModel):
    
    def __init__(self, tracks_info, user_info, config, top_k: int=100):
        super(RecRunner, self).__init__()
        """
        :param top_k: numbers of recommendation to return for each user. Defaults to 20.
        """

        self.top_k = top_k
        self.config = config
        config['user_numerical_num'] = 3

        self.user_info = self._convert_user_info(user_info)

        # model setting
        self.device = torch.device("cuda:{}".format(config['gpu_index'])) if torch.cuda.is_available() else torch.device("cpu")
        # self.device = torch.device("cpu")

    def _convert_user_info(self, user_info):
        user_info['gender'] = user_info['gender'].fillna(value='n')
        user_info['country'] = user_info['country'].fillna(value='UNKNOWN')

        codes_genders, uniques_genders = pd.factorize(user_info['gender'])
        user_info['converted_gender'] = codes_genders + 1              # reserve 0 for pad

        codes_country, uniques_country = pd.factorize(user_info['country'])
        user_info['converted_country'] = codes_country + 1              # reserve 0 for pad

        self.config['gender_num'] = len(uniques_genders) + 1
        self.config['country_num'] = len(uniques_country) + 1

        # discretize age
        bins = [-1, 0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 200]
        bin_labels = [_+1 for _ in range(len(bins)-1)]
        user_info['age_bin'] = pd.cut(user_info['age'], bins, labels=bin_labels, include_lowest=True)
        self.config['age_num'] = len(bin_labels) + 1

        return user_info

    def _convert_track_info(self, df: pd.DataFrame):
        # convert track to continuous index (0 is for PAD)
        self.track_list, self.invert_track_list = pd.factorize(df['track_id'])
        df['converted_track_id'] = self.track_list + 1              # reserve 0 for pad
        self.config['track_num'] = len(self.invert_track_list) + 1
        print("Track num: {}".format(self.config['track_num']))

        # self.artist_list, uniques_artist_list = pd.factorize(df['artist_id'])
        # df['converted_artist_id'] = self.artist_list + 1              # reserve 0 for pad
        # self.config['artist_num'] = len(uniques_artist_list) + 1
        # print("Artist num: {}".format(self.config['artist_num']))

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["hour"] = df["timestamp"].dt.hour

        return df

    def _prepare_train_data(self, df):
        # convert track id to feed into embedding layer
        # total: 820998 unique tracks
        self.df = self._convert_track_info(df)

        # prepare track bin
        bins = np.array([1, 10, 100, 1000])
        track_activity = df.groupby('converted_track_id', as_index=True, sort=False)[['user_track_count']].sum()
        track_activity['bin_index'] = np.digitize(track_activity.values.reshape(-1), bins)
        track_activity['bins'] = bins[track_activity['bin_index'].values - 1]
        # track_activity = track_activity[['track_id', 'bin_index']]

        self.track_bins = {}
        for key, value in track_activity['bin_index'].to_dict().items():
            if value not in self.track_bins.keys():
                self.track_bins[value] = [key]
            else:
                self.track_bins[value].append(key)

        # build dataloader
        data = RecDataset(df=self.df, user_info=self.user_info, mode='train', config=self.config)
        dataloader = DataLoader(data, batch_size=self.config['batch'], num_workers=4, pin_memory=False)
        return dataloader

    def _negative_sampling(self, logits, labels):
        class_num = logits.shape[1]
        candidates = torch.randint(low=0, high=logits.shape[1], size=(class_num-self.config['negative_samples'],))
        
        for mini_batch in range(len(labels)):
            label_logits = logits[mini_batch, labels[mini_batch]]
            logits[mini_batch, candidates] = float("-inf")
            logits[mini_batch, labels[mini_batch]] = label_logits

        # user x class_num, user
        # print(logits.shape, labels.shape, candidates.shape)
        return logits

    def _trainer(self, dataloader):
        pbar = tqdm(range(self.config['epochs']), desc='Epoch: ')
        train_loss = []
        self.recformer.train()
        for epoch in pbar:
            total_loss = 0
            for batch_index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                self.optimizer.zero_grad()
                sequences, genders, countrys, hours, ages, labels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device), data[4].to(self.device), data[5].to(self.device)
                logits = self.recformer(sequences=sequences, genders=genders, countrys=countrys, hours=hours, ages=ages)

                # remove pad and unmasked labels and tokens
                logits = logits[labels!=0]
                labels = labels[labels!=0]

                # logits = self._negative_sampling(logits, labels)

                ce_loss = self.criterion(logits, labels)
                ce_loss.backward()
                self.optimizer.step()
                loss = ce_loss.item()
                total_loss += loss
                pbar.set_description("Loss: {}".format(round(loss, 3)), refresh=True)

                if self.config['is_debug']:
                    if batch_index == 2:
                        break

                del loss, logits, sequences, genders, countrys, labels
            
            total_loss /= len(dataloader)
            train_loss.append(total_loss)

        return train_loss

    def train(self, train_df: pd.DataFrame, **kwargs):
        print("==== Start Preparing Training Data ====")
        train_dataloader = self._prepare_train_data(train_df)

        self.recformer = TransformerEncoder(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.recformer.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config['label_smoothing'])

        print("Total parameters:")
        print(sum(p.numel() for p in self.recformer.parameters() if p.requires_grad))
        print(self.config)

        print("==== Start Training ====")
        train_loss = self._trainer(train_dataloader)

        print(train_loss)

        if self.config['is_save']:
            if not os.path.exists(self.config['save_path']):
                os.makedirs(self.config['save_path'])
            torch.save(self.recformer.state_dict(), '{}{}.pt'.format(self.config['save_path'], 'model'))

    def _prepare_test_data(self, test_df):
        data = RecDataset(df=self.df, user_info=self.user_info, mode='test', config=self.config, test_df=test_df)
        # [TODO]: can batchify now, but has error in below codes
        dataloader = DataLoader(data, batch_size=1, num_workers=4)
        return dataloader

    def _predictor(self, dataloader):
        self.recformer.eval()
        with torch.no_grad():
            users, predictions = [], []
            for batch_index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                self.optimizer.zero_grad()
                user_id, sequences, genders, countrys, hours, ages = data[0].tolist(), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device), data[4].to(self.device), data[5].to(self.device)

                logits = self.recformer(sequences=sequences, genders=genders, countrys=countrys, hours=hours, ages=ages)

                # the last token is the predicted one
                logits = logits[:, -1]

                # # select indexes of each bin and select top k
                # invert_top_suggestions = None
                # for key, values in self.track_bins.items():
                #     topk_suggestions = torch.topk(input=logits[:, values], k=25, largest=True)[1] - 1      # 25 = 100 / 4bins
                #     if invert_top_suggestions is None:
                #         invert_top_suggestions = self.invert_track_list[topk_suggestions.cpu().detach().flatten().tolist()].values.tolist()
                #     else:
                #         invert_top_suggestions += self.invert_track_list[topk_suggestions.cpu().detach().flatten().tolist()].values.tolist()

                # # top k selection
                # indices_to_remove = logits < torch.topk(logits, self.top_k*2)[0][..., -1, None]
                # logits[indices_to_remove] = float("-inf")
                # p = torch.nn.functional.softmax(logits, dim=-1)
                # topk_suggestions = p.multinomial(num_samples=self.top_k, replacement=False).cpu().detach().flatten().tolist()

                # HR better but overall worse
                # # remove past tracks as -1e3
                # past_tracks = self.df.loc[self.df['user_id']==user_id[0], 'converted_track_id'].values.tolist()
                # logits[:, past_tracks] = -1e3

                topk_suggestions = (torch.topk(input=logits, k=self.top_k, largest=True)[1]-1).cpu().detach().flatten().tolist()
                invert_top_suggestions = self.invert_track_list[topk_suggestions].values.tolist()

                # minor_class = self.track_bins[1]
                # topk_suggestions = torch.topk(input=logits[:, minor_class], k=10, largest=True)[1] - 1
                # minor_top_suggestions = self.invert_track_list[topk_suggestions.cpu().detach().flatten().tolist()].values.tolist()
                # invert_top_suggestions[:10] = minor_top_suggestions

                # invert_top_suggestions = []
                # for suggestion in topk_suggestions:
                #     invert_top_suggestions.append(self.invert_track_list[suggestion-1])

                users.append(user_id)
                predictions.append(invert_top_suggestions)
        
        return users, predictions

    def predict(self, user_ids: pd.DataFrame):
        print("==== Start Preparing Testing Data ====")
        test_dataloader = self._prepare_test_data(user_ids['user_id'].values.tolist())

        print("==== Start Testing ====")
        users, predictions = self._predictor(test_dataloader)

        predictions = np.concatenate([np.array(users), np.array(predictions)], axis=1)
        predictions = pd.DataFrame(predictions, columns=['user_id', *[str(i) for i in range(self.top_k)]]).set_index('user_id')

        print(predictions.head(3))

        return predictions
