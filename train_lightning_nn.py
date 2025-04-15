import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

from model import Net
from dataset import TabularData
from utils import process_cat_cols,convert_cat_to_one_hot

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from watermark import watermark
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import time

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os

class RegressionModel(pl.LightningModule):
    def __init__(self, model, model_lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model=model
        self.model_lr=model_lr
        
        self.val_mae=MeanAbsoluteError()
        self.val_mse=MeanSquaredError()
        self.val_r2=R2Score()
        
        self.test_mae=MeanAbsoluteError()
        self.test_mse=MeanSquaredError()
        self.test_r2=R2Score()
        
    def forward(self, cat_feature_list, non_cat_features):
        return self.model(cat_feature_list, non_cat_features)
    
    def training_step(self, batch, batch_idx):
        features, labels=batch
        non_cat_features=features[:,:len(non_cat_cols)]
        features=features[:,len(non_cat_cols):]
        cat_feature_list=[]
        for col in cat_index_dict.keys():
            cat_feature_list+=[features[:,:len(cat_index_dict[col])]]
            features=features[:,len(cat_index_dict[col]):]  
        
        y_hat = self(cat_feature_list, non_cat_features)
        loss = nn.MSELoss()(y_hat, labels)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, labels=batch
        non_cat_features=features[:,:len(non_cat_cols)]
        features=features[:,len(non_cat_cols):]
        cat_feature_list=[]
        for col in cat_index_dict.keys():
            cat_feature_list+=[features[:,:len(cat_index_dict[col])]]
            features=features[:,len(cat_index_dict[col]):]  
        
        y_hat = self(cat_feature_list, non_cat_features)
        loss = nn.MSELoss()(y_hat, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_mae', self.val_mae(y_hat,labels), prog_bar=True, on_epoch=True)
        self.log('val_mse', self.val_mse(y_hat,labels), prog_bar=True, on_epoch=True)
        self.log('val_r2', self.val_r2(y_hat,labels), prog_bar=True, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        features, labels=batch
        non_cat_features=features[:,:len(non_cat_cols)]
        features=features[:,len(non_cat_cols):]
        cat_feature_list=[]
        for col in cat_index_dict.keys():
            cat_feature_list+=[features[:,:len(cat_index_dict[col])]]
            features=features[:,len(cat_index_dict[col]):]  
        
        y_hat = self(cat_feature_list, non_cat_features)
    
        self.log('test_mae', self.test_mae(y_hat,labels), prog_bar=True, on_epoch=True)
        self.log('test_mse', self.test_mse(y_hat,labels), prog_bar=True, on_epoch=True)
        self.log('test_r2', self.test_r2(y_hat,labels), prog_bar=True, on_epoch=True)
        
       
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.model_lr)

if __name__ == "__main__":
    print(watermark(packages="torch,pytorch_lightning,transformers", python=True), flush=True)
    print("Torch CUDA available?", torch.cuda.is_available(), flush=True)
    
    batch_size=32
    n_epochs=100
    num_workers=10
    model_lr=1e-3
    
    df=pd.read_csv('./Housing.csv')

    cat_cols=['mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']
    non_cat_cols=['area','bedrooms','bathrooms','stories']
    y_col=['price']

    df=df[cat_cols+non_cat_cols+y_col]
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        df[cat_cols+non_cat_cols], df[y_col], 
        test_size=0.1,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.1,
        random_state=42
    )
    
    df_train=pd.concat([X_train.reset_index(drop=True),y_train.reset_index(drop=True)], axis=1)
    df_train['dtype']='train'
    
    df_val=pd.concat([X_val.reset_index(drop=True),y_val.reset_index(drop=True)], axis=1)
    df_val['dtype']='val'
    
    df_test=pd.concat([X_test.reset_index(drop=True),y_test.reset_index(drop=True)], axis=1)
    df_test['dtype']='test'
    
    df=pd.concat([df_train,df_val,df_test]).reset_index(drop=True)
    
    # non_cat_df=df[non_cat_cols]
    scaler=MinMaxScaler()
    scaler.fit(df_train[non_cat_cols])
    non_cat_df=pd.concat([pd.DataFrame(scaler.transform(x), columns=non_cat_cols) for x in [df_train[non_cat_cols],df_val[non_cat_cols],df_test[non_cat_cols]]]).reset_index(drop=True)
    cat_df=df[cat_cols]
    y=df[y_col]
    dtype=df['dtype']

    res_df, cat_index_dict = process_cat_cols(df, cat_cols)

    df = pd.concat([non_cat_df, res_df, dtype, y], axis=1)
    df = df[df['dtype']==dtype]
    
    # Create train/val splits
    train_dataset = TabularData(df,dtype='train', cat_cols=cat_cols, non_cat_cols=non_cat_cols, y_col=y_col)
    val_dataset = TabularData(df,dtype='val', cat_cols=cat_cols, non_cat_cols=non_cat_cols, y_col=y_col)
    test_dataset = TabularData(df,dtype='test', cat_cols=cat_cols, non_cat_cols=non_cat_cols, y_col=y_col)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    # Initialize model and trainer
    cat_index_dict=cat_index_dict
    embedding_dim=6
    num_numerical_features=len(non_cat_cols)
    hidden_dim=sum([len(cat_index_dict[col]) for col in cat_index_dict.keys()])*embedding_dim+num_numerical_features
    output_dim=1

    model=Net(cat_index_dict, embedding_dim, hidden_dim, output_dim)
    
    lightning_model = RegressionModel(model, model_lr=model_lr)
    
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="min", monitor="val_loss", filename='{epoch}-{val_loss:.2f}-{step:.2f}'),  # save top 1 model
        # ModelCheckpoint(save_last=True, filename='{epoch}-{val_bleu:.2f}-{step:.2f}'),  # save last model
        EarlyStopping(monitor="val_loss", min_delta=0.000, patience=12, verbose=False, mode="min"),
        # StochasticWeightAveraging(swa_lrs=1e-2)
    ]
    
    csv_logger = CSVLogger(save_dir="./", name=f"regression_model_filtered_data")
    
    start = time.time()
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        callbacks=callbacks,
        logger=csv_logger,
    )
    
    # Train the model
    trainer.fit(lightning_model, train_loader, val_loader)
    
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    test_bleu1 = trainer.test(lightning_model, test_loader, ckpt_path="best")
    # test_bleu2 = trainer.test(lightning_model, test_loader2, ckpt_path="best")

    with open(os.path.join(trainer.logger.log_dir, "outputs.txt"), "w") as f:
        f.write((f"Time elapsed {elapsed/60:.2f} min\n"))
        f.write(f"Test Accuracy: {test_bleu1}\n")