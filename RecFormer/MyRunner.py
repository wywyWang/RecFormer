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
    
    def __init__(self, tracks_info, config, top_k: int=100):
        super(RecRunner, self).__init__()
        """
        :param top_k: numbers of recommendation to return for each user. Defaults to 20.
        """
        self.top_k = top_k

        self.tracks_info = tracks_info
        # convert track to continuous index (0 is for PAD)
        track_list = tracks_info.index.to_list()
        self.track_list = {value: index+1 for index, value in enumerate(track_list)}

        self.config = config
        self.mappings = None
        
        # model setting
        self.device = torch.device("cuda:{}".format(config['gpu_index'])) if torch.cuda.is_available() else torch.device("cpu")
        self.recformer = TransformerEncoder(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.recformer.parameters(), lr=config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()

    def _convert_track_id(self, df: pd.DataFrame):
        df_dt = dt.Frame(df)
        converted_track_list = []
        for index in tqdm(range(df_dt[:, 'track_id'].shape[0])):
            track = df_dt[index, 'track_id']
            converted_track_list.append(self.track_list[track])
        df_dt.cbind(dt.Frame(converted_track_id=converted_track_list))

        return df_dt.to_pandas()

    def _prepare_data(self, df):
        # convert track id to feed into embedding layer
        # total: 820998 unique tracks
        df = self._convert_track_id(df)

        # build dataloader
        data = RecDataset(df=df, mode='train', config=self.config)
        dataloader = DataLoader(data, batch_size=self.config['batch'])
        return dataloader

    def _trainer(self, dataloader):
        pbar = tqdm(range(self.config['epochs']), desc='Epoch: ')
        train_loss = []
        for epoch in pbar:
            self.recformer.train()
            total_loss = 0
            for batch_index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                self.optimizer.zero_grad()
                sequences, labels = data[0].to(self.device), data[1].to(self.device)

                logits = self.recformer(sequences)

                # remove pad and unmasked labels and tokens
                logits = logits[labels!=0]
                labels = labels[labels!=0]

                ce_loss = self.criterion(logits, labels)
                total_loss += ce_loss.item()
                ce_loss.backward()
                self.optimizer.step()

                if batch_index % 100 == 0:
                    print(ce_loss.item())
            
            total_loss /= len(dataloader)
            train_loss.append(total_loss)
        return train_loss

    def train(self, train_df: pd.DataFrame, **kwargs):
        print("==== Start Preparing Training Data ====")
        train_dataloader = self._prepare_data(train_df)

        print("==== Start Training ====")
        train_loss = self._trainer(train_dataloader)

        print(train_loss)
        1/0

        # user_tracks = pd.DataFrame(p)
        
        # # we sample 40 songs for each user. This will be used at runtime to build a user vector
        # user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 

        # # this dictionary maps users to the songs:
        # # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}
        # self.mappings = user_tracks.T.to_dict()

    def predict(self, user_ids: pd.DataFrame):

        user_ids = user_ids.copy()
        k = self.top_k

       
        pbar = tqdm(total=len(user_ids), position=0)
        
        predictions = []
        
        # probably not the fastest way to do this
        for user in user_ids["user_id"]:
          
          	# for each user we get their sample tracks
            user_tracks = self.mappings[user]["track_id_sampled"]
            
            # average to get user embedding
            get_user_embedding = np.mean([self.mymodel.wv[t] for t in user_tracks], axis=0)
            
            
            # we need to filter out stuff from the user history. We don't want to suggest to the user 
            # something they have already listened to
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k

            # let's predict the tracks
            user_predictions = [k[0] for k in self.mymodel.wv.most_similar(positive=[get_user_embedding], 
                                                                           topn=max_number_of_returned_items)]
            # let's filter out songs the user has already listened to
            user_predictions = list(filter(lambda x: x not in 
                                           self.mappings[user]["track_id"], user_predictions))[0:self.top_k]
            
            # append to the return list
            predictions.append(user_predictions)

            pbar.update(1)
        pbar.close()
        
        # lil trick to reconstruct a dataframe that has user ids as index and the predictions as columns
        # This is a very important part! consistency in how you return the results is fundamental for the 
        # evaluation
     
        users = user_ids["user_id"].values.reshape(-1, 1)
        predictions = np.concatenate([users, np.array(predictions)], axis=1)
        return pd.DataFrame(predictions, columns=['user_id', *[str(i) for i in range(k)]]).set_index('user_id')