import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_file_path = "data/train.csv"
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

features = ['MSSubClass',
'LotArea',
'OverallQual',
'OverallCond',
'YearBuilt',
'LowQualFinSF',
'EnclosedPorch',
'YearRemodAdd',
'1stFlrSF',
'2ndFlrSF',
'GrLivArea',
'WoodDeckSF',
'FullBath',
'HalfBath',
'BedroomAbvGr',
'KitchenAbvGr',
'TotRmsAbvGrd',
'EnclosedPorch',
'3SsnPorch',
'ScreenPorch',
'PoolArea',
'YrSold']
# New features added are 'Neighborhood','YearRemodAdd','Fence','PoolArea','PoolQC','GarageQual','GarageCond','GarageCars','GarageArea','KitchenQual'
# Removed Neighborhood, could not convert string to float?
# Removed quality features and condition features, having issue converting string to float
# Removed fence, worked, got a mae of 22306 which is slightly worse than my first submission
# Adding the OverallQual and OverallCond features since they are numbers
# Model improved by 3000 to 19010
# Testing removing garage area to see if there is a change
# Model improved by 500 to 18517
# Testing removing garage cars to see if there is a change
# Model got worse by 400, keeping garage cars
# Removing poolarea to see if there is a change
# Model barely got worse, keeping pool area
# Adding MiscVal to see if there is a change
# Model barely got worse
# Adding all potential columns that won't throw errors
# Model improved to 17906
# Removing fireplaces to see if there is a change
# Model improved by 50
# Removing month sold to see if there is a change
# Model improved by 160
# Removing screen porch to see if there is a change
# Model worsened by 150
# Removing 3ssn porch to see if there is a change
# Model worsened by 80
# Removing misc val to see if there is a change
# Misc val did not affect change
# Removing year sold to see if there is a change
# Model worsened by 40
# Removing WoodDeckSF to see if there is a change
# Model worsened by 110
# Removing OpenPorchSF to see if there is a change
# Model improved by 90
# Removing LowQualFinSF to see if there is a change
# Model worsened by 14
# Removing EnclosedPorch to see if there is a change
# Model worsened by 110


X = home_data[features]

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X,train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions,val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))