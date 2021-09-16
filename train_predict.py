import numpy as np
import pandas as pd
import lightgbm as lgb
import random
import warnings
warnings.filterwarnings('ignore')

from src.features import generate_date_features, get_cross_tkp_metrics, generate_lag_features
from src.util import seed_everything, mae, mape, read_data, create_folds


# Default parameters
TARGET = ['case', 'unit_cost']
SEED = 2021
categorical_features = [
    'kddati2',
    'tkp',
    'id',
    'month',
    'year'
]
rs_features = [
    'a', 'b', 'c', 'cb', 'd', 'ds', 'gd', 'hd', 
    'i1', 'i2', 'i3', 'i4', 
    'kb', 'kc', 'kg', 'ki', 'kj', 'kk', 'kl', 'km', 'ko', 'kp', 'kt', 'ku', 
    's', 'sa', 'sb', 'sc', 'sd'
]
remove_features = ['tglpelayanan','row_id','cat','kfold']
numerical_features = ['peserta'] + TARGET + ['case2', 'unit_cost2']
FOLD = 5


# Preprocessing
df = read_data(dir='data')
df['id'] = df['kddati2'].astype(str) + '-' + df['tkp'].astype(str)
df['rs_total'] = df[rs_features].sum(axis=1)

# Features engineering
df = generate_date_features(df)
df = get_cross_tkp_metrics(df)
df = generate_lag_features(df, features=numerical_features)

# Remove unused variables
df = df.drop(columns=['tglpelayanan_lag_1','tglpelayanan_lag_-1'])

# Create folds
df = create_folds(df, cv_split=5, seed=SEED)

# Fill NA with lag/lead
nlag_features = [col for col in df.columns if 'lag_1' in col]
nlead_features = [col for col in df.columns if 'lag_-1' in col]
df[nlag_features] = df.groupby(['id'])[nlag_features].bfill()
df[nlead_features] = df.groupby(['id'])[nlead_features].ffill()


# Model
def process_train_lgb(
    df,
    target=TARGET[0],
    seed=SEED,
    verbose=500,
    validation=True,
    remove_additional_features=[],
    kfold=0,
    use_log=False
):
    df = df.copy()
    df_test = df[df['cat'] == 'Test']
    df = df[df['cat'] == 'Train']
    local_params = lgb_params.copy()

    if use_log:
        df[target] = np.log1p(df[target])

    # Categorical features
    for col in categorical_features:
        try:
            df[col] = df[col].astype('category')
            df_test[col] = df_test[col].astype('category')
        except:
            pass

    # All features
    remove_additional_features_selected = list(set(remove_additional_features) & set(df.columns.tolist()))  
    all_features = [col for col in list(df) if col not in (remove_features + remove_additional_features + TARGET)]

    print(all_features)

    # Split train and valid
    if validation:
        train_data = lgb.Dataset(df[df['kfold'] != kfold][all_features], label=df[df['kfold'] != kfold][target])
        valid_data = lgb.Dataset(df[df['kfold'] == kfold][all_features], label=df[df['kfold'] == kfold][target])
    else:
        train_data = lgb.Dataset(df[all_features], label=df[target])

    # Training process
    seed_everything()
    if validation:
        estimator = lgb.train(local_params,
                              train_data,
                              valid_sets = [train_data,valid_data],
                              verbose_eval = verbose)
        temp_df = df[df['kfold'] == kfold]

    else:
        if 'early_stopping_rounds' in local_params: 
            del local_params['early_stopping_rounds']
        estimator = lgb.train(local_params,
                              train_data,
                              valid_sets = [train_data],
                              verbose_eval = verbose)
        temp_df = df_test

    temp_df['predict_' + target] = estimator.predict(temp_df[all_features])

    if use_log:
        temp_df['predict_' + target] = np.expm1(temp_df['predict_' + target])
        temp_df[target] = np.expm1(temp_df[target])


    temp_df['predict_' + target] = temp_df['predict_' + target].clip(lower=1)

    if validation:
        print('MAPE CV-{} is {:.2f}%'.format(kfold+1, mape(temp_df[target], temp_df['predict_' + target])))
        print('MAE CV-{} is {:.2f}'.format(kfold+1, mae(temp_df[target], temp_df['predict_' + target])))

    temp_df = temp_df[['row_id','tglpelayanan','kddati2','tkp',target,'predict_' + target,'kfold']]

    return estimator, temp_df


lgb_params = {'boosting_type': 'gbdt', 
              'objective': 'mape',
              'metric': ['mape'], 
              'learning_rate': 0.1,      
              'subsample': 0.9,       
              'subsample_freq': 1,     
              'num_leaves': 255,            
              'min_data_in_leaf': 255, 
              'feature_fraction': 0.9,
              'n_estimators': 1000,   
              'early_stopping_rounds': 100,
              'seed': 20000,
              'verbose': -1}

case_mdl, case_test_df = process_train_lgb(
    df,
    remove_additional_features=rs_features,
    validation=False
)

cost_test_df = df[df['cat'] == 'Test'][['row_id','unit_cost_lag_-1','unit_cost_lag_1']].copy()
cost_test_df['predict_unit_cost'] = (cost_test_df['unit_cost_lag_-1'] + cost_test_df['unit_cost_lag_1']) / 2
cost_test_df = cost_test_df.drop(['unit_cost_lag_-1','unit_cost_lag_1'], axis=1)

sub = pd.concat([case_test_df[['row_id','predict_case']], cost_test_df[['predict_unit_cost']]], axis=1).sort_values('row_id')
sub.to_csv('submission/tahap2_case_cost_prediction_V2.csv', index=False)

case_mdl.save_model('submission/tahap2_case_model_V2.mdl')

