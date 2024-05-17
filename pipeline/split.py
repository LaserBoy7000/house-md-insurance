import pandas as pd
import static.columns as columns
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/data.csv", index_col=False)

train, test = train_test_split(df, train_size=0.9)

train.to_csv("./data/train.csv", index=False)
test[columns.X_columns].to_csv("./data/new.csv", index=False)