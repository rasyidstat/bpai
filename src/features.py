import numpy as np
import pandas as pd

def generate_date_features(df):
    df['tglpelayanan'] = pd.to_datetime(df['tglpelayanan'])
    df['month'] = df['tglpelayanan'].dt.month
    df['year'] = df['tglpelayanan'].dt.year

    return df

def get_cross_tkp_metrics(df):
    df_temp = df[['tglpelayanan','kddati2','tkp','case','unit_cost']].copy()
    df_temp['tkp'] = np.where(df_temp['tkp'] == 30, 40, 30)
    df_temp = df_temp.rename(columns={'case':'case2', 'unit_cost':'unit_cost2'})
    df = df.merge(df_temp,how='left')

    return df

def generate_lag_features(df, features):
    df = df.assign(**{
            '{}_lag_{}'.format(col, l): df.groupby(['id'])[col].transform(lambda x: x.shift(l))
            for l in [1,-1]
            for col in features + ['tglpelayanan']
        })
    lag_features = [col for col in df.columns if 'lag' in col]
    nlag_features = [col for col in lag_features if 'lag_1' in col and 'tglpelayanan' not in col and 'peserta' not in col]
    nlead_features = [col for col in lag_features if 'lag_-1' in col and 'tglpelayanan' not in col and 'peserta' not in col]

    for i, col in enumerate(['case2']):
        df['ds_' + col] = np.where(df[col].isnull(), None, df['tglpelayanan'].astype(str))
        df['ds_' + col] = df.groupby(['id'])['ds_' + col].ffill()
        df['ds_' + col] = np.round(((df['tglpelayanan'] - pd.to_datetime(df['ds_' + col]))/np.timedelta64(1, 'M')))
    df[['case2','unit_cost2']] = df.groupby(['id'])['case2','unit_cost2'].ffill()

    for i, col in enumerate(['case_lag_1','case2_lag_1']):
        df['ds_' + col] = np.where(df[col].isnull(), None, df['tglpelayanan_lag_1'].astype(str))
        df['ds_' + col] = df.groupby(['id'])['ds_' + col].ffill()
        df['ds_' + col] = np.round(((df['tglpelayanan'] - pd.to_datetime(df['ds_' + col]))/np.timedelta64(1, 'M')))
    df[nlag_features] = df.groupby(['id'])[nlag_features].ffill()

    for i, col in enumerate(['case_lag_-1','case2_lag_-1']):
        df['ds_' + col] = np.where(df[col].isnull(), None, df['tglpelayanan_lag_-1'].astype(str))
        df['ds_' + col] = df.groupby(['id'])['ds_' + col].bfill()
        df['ds_' + col] = np.round(((df['tglpelayanan'] - pd.to_datetime(df['ds_' + col]))/np.timedelta64(1, 'M')))
    df[nlead_features] = df.groupby(['id'])[nlead_features].bfill()

    return df