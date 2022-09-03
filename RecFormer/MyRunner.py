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
import datatable as dt


class RecRunner(RecModel):
    
    def __init__(self, tracks_info, user_info, config, top_k: int=100):
        super(RecRunner, self).__init__()
        """
        :param top_k: numbers of recommendation to return for each user. Defaults to 20.
        """

        self.top_k = top_k
        self.config = config

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

        return user_info

    def _convert_track_info(self, df: pd.DataFrame):
        # convert track to continuous index (0 is for PAD)
        track_list = df['track_id'].unique().tolist()
        self.track_list = {value: index+1 for index, value in enumerate(track_list)}
        self.invert_track_list = {index+1: value for index, value in enumerate(track_list)}
        self.config['track_num'] = len(track_list)

        # convert artist to continuous index (0 is for PAD)
        artist_list = df['artist_id'].unique().tolist()
        self.artist_list = {value: index+1 for index, value in enumerate(artist_list)}
        self.invert_artist_list = {index+1: value for index, value in enumerate(artist_list)}
        self.config['artist_num'] = len(artist_list)

        df_dt = dt.Frame(df)
        converted_track_list = []
        for index in tqdm(range(df_dt[:, 'track_id'].shape[0])):
            track = df_dt[index, 'track_id']
            converted_track_list.append(self.track_list[track])
        df_dt.cbind(dt.Frame(converted_track_id=converted_track_list))

        converted_artist_list = []
        for index in tqdm(range(df_dt[:, 'artist_id'].shape[0])):
            artist = df_dt[index, 'artist_id']
            converted_artist_list.append(self.artist_list[artist])
        df_dt.cbind(dt.Frame(converted_artist_id=converted_artist_list))

        return df_dt.to_pandas()

    def _prepare_train_data(self, df):
        # convert track id to feed into embedding layer
        # total: 820998 unique tracks
        self.df = self._convert_track_info(df)

        # build dataloader
        data = RecDataset(df=self.df, user_info=self.user_info, mode='train', config=self.config)
        dataloader = DataLoader(data, batch_size=self.config['batch'])
        return dataloader

    def _trainer(self, dataloader):
        pbar = tqdm(range(self.config['epochs']), desc='Epoch: ')
        train_loss = []
        self.recformer.train()
        for epoch in pbar:
            total_loss = 0
            for batch_index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                self.optimizer.zero_grad()
                sequences, artists, genders, countrys, labels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device), data[4].to(self.device)

                logits = self.recformer(sequences=sequences, artists=artists, genders=genders, countrys=countrys)

                # remove pad and unmasked labels and tokens
                logits = logits[labels!=0]
                labels = labels[labels!=0]

                # print(torch.topk(input=logits[0], k=3, largest=True)[1])
                # print(labels[0])

                ce_loss = self.criterion(logits, labels)
                loss = ce_loss.item()
                total_loss += loss
                ce_loss.backward()
                self.optimizer.step()
                pbar.set_description("Loss: {}".format(round(loss, 3)), refresh=True)

                # if batch_index % 100 == 0:
                #     print()
                #     print("Current loss: {}".format(ce_loss.item()))

                if self.config['is_debug']:
                    if batch_index == 2:
                        break
            
            total_loss /= len(dataloader)
            train_loss.append(total_loss)
        return train_loss

    def train(self, train_df: pd.DataFrame, **kwargs):
        print("==== Start Preparing Training Data ====")
        train_dataloader = self._prepare_train_data(train_df)

        self.recformer = TransformerEncoder(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.recformer.parameters(), lr=self.config['learning_rate'], weight_decay=1e-2)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.3)

        print("==== Start Training ====")
        train_loss = self._trainer(train_dataloader)

        print(train_loss)

        if self.config['is_save']:
            if not os.path.exists(config['save_path']):
                os.makedirs(config['save_path'])
            torch.save(self.recformer.state_dict(), '{}{}.pt'.format(config['save_path'], 'model'))

    def _prepare_test_data(self, test_df):
        data = RecDataset(df=self.df, user_info=self.user_info, mode='test', config=self.config, test_df=test_df)
        # [TODO]: can batchify now, but has error in below codes
        dataloader = DataLoader(data, batch_size=1)
        return dataloader

    def _predictor(self, dataloader):
        self.recformer.eval()
        with torch.no_grad():
            users, predictions = [], []
            for batch_index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                self.optimizer.zero_grad()
                user_id, sequences, artists, genders, countrys = data[0].tolist(), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device), data[4].to(self.device)

                logits = self.recformer(sequences=sequences, artists=artists, genders=genders, countrys=countrys)

                # the last token is the predicted one
                logits = logits[:, -1]

                # # remove past tracks as -1e3
                # # decrease performance
                # past_tracks = self.df.loc[self.df['user_id']==user_id[0], 'converted_track_id'].values.tolist()
                # logits[:, past_tracks] = -1e3

                topk_suggestions = torch.topk(input=logits, k=self.top_k, largest=True)[1].cpu().detach().flatten().tolist()

                invert_top_suggestions = []
                for suggestion in topk_suggestions:
                    invert_top_suggestions.append(self.invert_track_list[suggestion])

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