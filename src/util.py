import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

def mape(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mae(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)))

def read_data(dir):
    train = pd.read_csv(dir + '/case_cost_prediction_train.csv')
    test = pd.read_csv(dir + '/case_cost_prediction_val.csv')
    df = pd.concat([train.assign(cat = 'Train'), test.assign(cat = 'Test')])
    df = df.sort_values(['kddati2','tkp','tglpelayanan'])
    return df

def create_folds(df, cv_split, seed):
    df['kfold'] = -1
    kf = StratifiedKFold(n_splits=cv_split, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, y=df['id'])):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    return df