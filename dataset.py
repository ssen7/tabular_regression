from torch.utils.data import Dataset
import torch
import pandas as pd

class TabularData(Dataset):

    def __init__(self, df:pd.DataFrame, dtype:str='train', cat_cols:list=[], non_cat_cols:list=[], y_col:list=[]):
        super().__init__()

        self.dtype=dtype
        self.cat_cols=cat_cols
        self.non_cat_cols=non_cat_cols
        self.y_col=y_col
        self.df=df[df['dtype']==dtype]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        features=torch.FloatTensor(self.df.drop(columns=[self.y_col[0], 'dtype']).iloc[index])
        label=torch.FloatTensor(self.df[self.y_col].iloc[index])
        
        return features, label