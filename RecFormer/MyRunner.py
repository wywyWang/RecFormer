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

        return user_info

    def _convert_track_info(self, df: pd.DataFrame):
        # convert track to continuous index (0 is for PAD)
        self.track_list, self.invert_track_list = pd.factorize(df['track_id'])
        df['converted_track_id'] = self.track_list + 1              # reserve 0 for pad
        self.config['track_num'] = len(self.invert_track_list) + 1
        print("Track num: {}".format(self.config['track_num']))

        self.artist_list, uniques_artist_list = pd.factorize(df['artist_id'])
        df['converted_artist_id'] = self.artist_list + 1              # reserve 0 for pad
        self.config['artist_num'] = len(uniques_artist_list) + 1
        print("Artist num: {}".format(self.config['artist_num']))

        return df

    def _prepare_train_data(self, df):
        # convert track id to feed into embedding layer
        # total: 820998 unique tracks
        self.df = self._convert_track_info(df)

        # build dataloader
        data = RecDataset(df=self.df, user_info=self.user_info, mode='train', config=self.config)
        dataloader = DataLoader(data, batch_size=self.config['batch'])
        return dataloader

    def _negative_sampling(self, logits, labels):
        sampling_logits = None
        class_num = logits.shape[1]
        candidates = torch.randint(low=0, high=logits.shape[1], size=(class_num-self.config['negative_samples'],))
        
        for mini_batch in range(len(labels)):
            label_logits = logits[mini_batch, labels[mini_batch]]
            logits[mini_batch, candidates] = -1e6
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
                sequences, artists, genders, countrys, novelty_artists, labels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device), data[4:7], data[7].to(self.device)
                logits = self.recformer(sequences=sequences, artists=artists, genders=genders, countrys=countrys, novelty_artists=novelty_artists)

                # if epoch == 3:
                #     print(sequences[0])
                #     print(labels[0])
                #     print(logits[0])
                #     print(torch.topk(input=logits[0], k=10, largest=True)[1])
                #     1/0

                # if batch_index == 0:
                #     # print(logits[0])
                #     # print(torch.topk(input=logits[0], k=10, largest=True)[1][0:2])
                #     print(sequences[0])
                #     print(labels[0])

                # remove pad and unmasked labels and tokens
                logits = logits[labels!=0]
                labels = labels[labels!=0]

                # logits = self._negative_sampling(logits, labels)

                # # print(torch.topk(input=logits[0], k=3, largest=True)[1])
                # print(logits)
                # print(labels)

                ce_loss = self.criterion(logits, labels)
                ce_loss.backward()
                self.optimizer.step()
                loss = ce_loss.item()
                total_loss += loss
                pbar.set_description("Loss: {}".format(round(loss, 3)), refresh=True)

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
        self.optimizer = torch.optim.AdamW(self.recformer.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config['label_smoothing'])

        print("Total parameters:")
        print(sum(p.numel() for p in self.recformer.parameters() if p.requires_grad))
        print(self.config)

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
                user_id, sequences, artists, genders, countrys, novelty_artists = data[0].tolist(), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device), data[4].to(self.device), data[5:8]

                logits = self.recformer(sequences=sequences, artists=artists, genders=genders, countrys=countrys, novelty_artists=novelty_artists)

                # the last token is the predicted one
                logits = logits[:, -1]

                # # remove past tracks as -1e3
                # # decrease performance
                # past_tracks = self.df.loc[self.df['user_id']==user_id[0], 'converted_track_id'].values.tolist()
                # logits[:, past_tracks] = -1e3

                topk_suggestions = torch.topk(input=logits, k=self.top_k, largest=True)[1].cpu().detach().flatten().tolist()

                invert_top_suggestions = []
                for suggestion in topk_suggestions:
                    invert_top_suggestions.append(self.invert_track_list[suggestion-1])

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