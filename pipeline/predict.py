import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Custom files

import static.columns as columns

# Read new data

ds = pd.read_csv("./data/new.csv")


# Feature engineering

param_dict = pickle.load(open('./models/param_dict.pickle', 'rb'))

# Outlier Engineering
for column in columns.numerical:
    ds[column] = ds[column].astype(float)
    ds = ds[~ np.where(ds[column] > param_dict['upper_lower_limits'][column+'_upper_limit'], True,
                       np.where(ds[column] < param_dict['upper_lower_limits'][column+'_lower_limit'], True, False))]

# Encoding

for col in columns.ohe_cols:
    ds[col] = ds[col].astype('str')

ohe = param_dict['ohe_encoder']
ohe_output = ohe.transform(ds[columns.ohe_cols])
ohe_output = pd.DataFrame(ohe_output, columns=ohe.get_feature_names_out(columns.ohe_cols))
ds = ds.reset_index(drop=True)
ds = ds.drop(columns.ohe_cols, axis=1)
ds = pd.concat([ds, ohe_output], axis=1)

for column in columns.oe_cols:
    ds[column] = ds[column].map(param_dict['ordinal_mapping'][column])

# Standartizarion

ds_scaled = param_dict['data_scaler'].transform(ds[columns.numerical])
ds = ds.drop(columns.numerical, axis=1)
ds = pd.concat([ds, pd.DataFrame(ds_scaled, columns=columns.numerical)], axis=1)

# Load the model and predict

cl = pickle.load(open('./models/model.sav', 'rb'))

y_pred = cl.predict(ds)
ds['charges'] = y_pred
ds.to_csv('./data/results.csv')