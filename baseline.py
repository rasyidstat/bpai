# Baseline regression

# Main package
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
categorical_features_2 = [
    'a', 'b', 'c', 'cb', 'd', 'ds', 'gd', 'hd', 
    'i1', 'i2', 'i3', 'i4', 
    'kb', 'kc', 'kg', 'ki', 'kj', 'kk', 'kl', 'km', 'ko', 'kp', 'kt', 'ku', 
    's', 'sa', 'sb', 'sc', 'sd'
]
remove_features = ['tglpelayanan','row_id','cat','kfold']
numerical_features = ['peserta'] + TARGET + ['case2', 'unit_cost2']

# Preprocessing
df = read_data(dir='data')
df['id'] = df['kddati2'].astype(str) + '-' + df['tkp'].astype(str)

# Features engineering
df = generate_date_features(df)
df = get_cross_tkp_metrics(df)
df = generate_lag_features(df, features=numerical_features)

# Remove unused variables
df = df.drop(columns=['tglpelayanan_lag_1','tglpelayanan_lag_-1'])

# Create folds
df = create_folds(df, cv_split=5, seed=SEED)


def process_train_lgb(
    df,
    target=TARGET[0],
    seed=SEED,
    verbose=500,
    validation=True,
    remove_additional_features=[],
    kfold=0
):
    df = df.copy()
    df_test = df[df['cat'] == 'Test']
    df = df[df['cat'] == 'Train']
    local_params = lgb_params.copy()

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
                              valid_sets = [train_data, valid_data],
                              verbose_eval = verbose)
        temp_df = df[df['kfold'] == kfold]
        temp_df['predict_' + target] = estimator.predict(temp_df[all_features])

        print('MAPE CV-{} is {:.2f}%'.format(kfold+1, mape(temp_df[target], temp_df['predict_' + target])))
        print('MAE CV-{} is {:.2f}'.format(kfold+1, mae(temp_df[target], temp_df['predict_' + target])))
    else:
        if 'early_stopping_rounds' in local_params: 
            del local_params['early_stopping_rounds']
        estimator = lgb.train(local_params,
                              train_data,
                              valid_sets = [train_data],
                              verbose_eval = verbose)
        temp_df = df_test
        temp_df['predict_' + target] = estimator.predict(temp_df[all_features])

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
              'feature_fraction': 0.95,
              'n_estimators': 10000,   
              'early_stopping_rounds': 100,
              'seed': SEED,
              'verbose': -1}
              
# Cross-Validation
val_df_all = pd.DataFrame({})
for fold in range(5):
    mdl, val_df = process_train_lgb(
        df,
        remove_additional_features=categorical_features_2,
        kfold=fold
    )
    val_df_all = val_df_all.append(val_df)
print('MAPE is {:.2f}%'.format(mape(val_df_all[TARGET[0]], val_df_all['predict_' + TARGET[0]])))
print('MAE is {:.2f}'.format(mae(val_df_all[TARGET[0]], val_df_all['predict_' + TARGET[0]])))

case_mdl, case_test_df = process_train_lgb(
    df,
    remove_additional_features=categorical_features_2,
    validation=False
)

lgb_params = {'boosting_type': 'gbdt', 
              'objective': 'regression',
              'metric': ['mae'], 
              'learning_rate': 0.025,      
              'subsample': 0.9,       
              'subsample_freq': 1,     
              'num_leaves': 255,            
              'min_data_in_leaf': 500, 
              'feature_fraction': 0.95,
              'n_estimators': 750,   
              'seed': SEED,
              'verbose': -1}
              
# Cross-Validation
mdl, val_df = process_train_lgb(
    df[['kddati2','tkp','unit_cost_lag_1','unit_cost_lag_-1','cat','id','unit_cost','row_id','tglpelayanan','kfold']],
    remove_additional_features=categorical_features_2,
    kfold=0,
    target=TARGET[1]
)

cost_mdl, cost_test_df = process_train_lgb(
    df[['kddati2','tkp','unit_cost_lag_1','unit_cost_lag_-1','cat','id','unit_cost','row_id','tglpelayanan','kfold']],
    remove_additional_features=categorical_features_2,
    validation=False,
    target=TARGET[1]
)

sub = pd.concat([case_test_df[['row_id','predict_case']], cost_test_df[['predict_unit_cost']]], axis=1).sort_values('row_id')
# sub.to_csv('submission/tahap2_case_cost_prediction.csv', index=False)