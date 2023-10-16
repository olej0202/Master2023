import sys
import os

package_path="C:\\Users\\Olej0\\OneDrive\\Dokumenter\\TFT_Model\\pytorch-forecasting"



# Check if the directory exists
if os.path.exists(package_path):
    # Add the package path to sys.path
    sys.path.append(package_path)


print(sys.path)

import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss,MultiHorizonMetric
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import numpy as np
df=pd.read_csv("Final_Dataset.csv").iloc[:,1:]

"""Energy_df=pd.read_csv("FINAL_ENERGY_PRODUCTION_DAILY").iloc[:,1:]
Electricity_df=pd.read_csv("FINAL_electricity_spot").iloc[:,1:]
Electricity_df"""
targets=df.iloc[:,-3:]
df=df.iloc[:,0:-7]
print(df)


print(targets["Gas_Price_EUR"].shift(-7).values[0:30])
df["Target"]=targets["Gas_Price_EUR"]
df=df.dropna()

print(df)

lags = [30,60,100] # Add lag variables for the previous 3 time steps

# Add lag variables to the DataFrame
for column in ['EU_clean_spark_spread',"Weighted_Average_Temperature","Net_Injection","Oil_Price","Wind_energy","Electricity_price","PINDUINDEXM",  "Google_data"]:
    for lag in lags:
        df[f'{column}_EMA{lag}'] = df[column]/df[column].ewm(span=lag, adjust=False).mean().values

merged_df = df

print(merged_df)
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df['month'] = merged_df['Date'].dt.month.astype(str)
merged_df['day_of_week'] = merged_df['Date'].dt.dayofweek.astype(str)
merged_df['Year'] = merged_df['Date'].dt.year.astype(str)
merged_df['Quarter'] = merged_df['Date'].dt.quarter.astype(str)
merged_df["time_idx"] = merged_df.index.tolist()
column_to_move = merged_df['time_idx']
merged_df = merged_df.drop('time_idx', axis=1)

# Step 3: Reinsert the column at the beginning
merged_df.insert(1, 'time_idx', column_to_move)
data=merged_df
data["constant"]=42



data=data.dropna()
print(len(np.array(data.columns[2:-5])[0]))

a1=np.array(data.columns[2:-5])
a2=np.array(data.columns[17:19])
print(a2)


cols=[]
for i in range(len(a1)):
  cols.append(a1[i])
  data[a1[i]] =pd.to_numeric(data[a1[i]])

cols1=[]
for i in range(len(a2)):
  cols1.append(a2[i])
  data[a2[i]] =pd.to_numeric(data[a2[i]])

cols1.append("time_idx")
cols1.append("Weighted_Average_Temperature")



cols = [item for item in cols if item not in cols1]


print(cols)
print(cols1)


max_prediction_length = 7
max_encoder_length = 65
training_cutoff = data["time_idx"].max() - max_prediction_length

lags={
    "Target":[1,2,3,4,5,6,7,8,9,10,28,29,30,31,60,364,365,366],


}

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Target",
    group_ids=["constant"],  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_encoder_length=max_encoder_length // 2,

    lags=lags,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_known_categoricals=["day_of_week", "month","Year","Quarter"],
    time_varying_known_reals=cols1,
    time_varying_unknown_reals=cols,
    target_normalizer=GroupNormalizer(
        groups=["constant"], transformation="softplus"
    ),

    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,





)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

batch_size = 32  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=8)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 5, num_workers=8)
data_df = training.data

# Display the first few rows of the data
print(data_df)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard
from lightning.pytorch import Trainer, seed_everything

seed_everything(42, workers=True)

trainer = pl.Trainer(
    max_epochs=17,
    enable_model_summary=True,
    gradient_clip_val=8,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
    deterministic=True,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.05,
    hidden_size=32,
    attention_head_size=8,
    dropout=0.25,
    hidden_continuous_size=16,
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    optimizer="Ranger",
    reduce_on_plateau_patience=4,
    lstm_layers=2,
    
    
    
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)