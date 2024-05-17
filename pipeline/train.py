import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Custom files

import static.model_best_hyperparameters as model_best_hyperparameters
import static.columns as columns

# Read train data

ds = pd.read_csv("./data/train.csv")

# Outlier Engineering

def find_skewed_boundaries(df, variable, distance):
    df[variable] = pd.to_numeric(df[variable],errors='coerce')
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary

upper_lower_limits = dict()
for column in columns.numerical:
    upper_lower_limits[column+'_upper_limit'], upper_lower_limits[column+'_lower_limit'] = find_skewed_boundaries(ds, column, 4)

for column in columns.numerical:
    ds = ds[~ np.where(ds[column] > upper_lower_limits[column+'_upper_limit'], True,
                       np.where(ds[column] < upper_lower_limits[column+'_lower_limit'], True, False))]
    
# One Hot encoding

ohe = OneHotEncoder(sparse_output=False)

for col in columns.ohe_cols:
    ds[col] = ds[col].astype('str')

ohe.fit(ds[columns.ohe_cols])
ohe_output = ohe.transform(ds[columns.ohe_cols])
ohe_output = pd.DataFrame(ohe_output)
ohe_output.columns = ohe.get_feature_names_out(columns.ohe_cols)
ds = ds.reset_index(drop=True)
ds = ds.drop(columns.ohe_cols, axis=1)
ds = pd.concat([ds, ohe_output], axis=1)

# Ordinal encoding

oe = dict()
for column in columns.oe_cols:
    mappings = ds[column].value_counts().index.tolist()
    mappings = {k: i for i, k in enumerate(mappings)}
    oe[column] = mappings
    ds[column] = ds[column].map(mappings)
    
# Sdandartization

scaler = StandardScaler()
scaler.fit(ds[columns.numerical])
ds_scaled = scaler.transform(ds[columns.numerical])
ds = ds.drop(columns.numerical, axis=1)
ds = pd.concat([ds, pd.DataFrame(ds_scaled, columns=columns.numerical)], axis=1)

# Save parameters 

param_dict = {
              'ordinal_mapping': oe,
              'ohe_encoder': ohe,
              'data_scaler': scaler,
              'upper_lower_limits': upper_lower_limits,
             }
with open('./models/param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# Define target and features columns

X = ds.drop(columns.Y_column, axis=1)
y = ds[columns.Y_column]

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.9)

cl = DecisionTreeRegressor(**model_best_hyperparameters.params)
cl.fit(X_train, y_train)
y_pred = cl.predict(X_test)
print('test set metrics: MAPE: ', metrics.mean_absolute_percentage_error(y_test, y_pred), ' MSE: ', metrics.mean_squared_error(y_test, y_pred))

# Save model

pickle.dump(cl, open('./models/model.sav', 'wb'))