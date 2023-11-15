import pandas as pd
from sklearn.tree import DecisionTreeRegressor
# save filepath to variable for easier access
file_path = 'train.csv'
# read the data and store data in DataFrame titled melbourne_data
file_data = pd.read_csv(file_path)
# print a summary of the data in Melbourne data
#test = melbourne_data.describe()

#file_data = file_data.dropna(axis=0)
print(file_data)
print(file_data.columns)
# y = prediction target
test = file_data.describe()
test = file_data.columns

y = file_data.SalePrice

data_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = file_data[data_features]

data_model = DecisionTreeRegressor(random_state=1)
data_model.fit(X, y)


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(data_model.predict(X.head()))