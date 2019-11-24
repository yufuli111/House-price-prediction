#import some necessary librairies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
import seaborn as sns
from sklearn.preprocessing import Imputer
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(train)
print(test)
print(train['SalePrice'].describe())

#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols1 = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols1].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols1.values, xticklabels=cols1.values)
plt.show()

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();

#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(22))

# delete outliers
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.show();
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# one hot
cols2=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
A=train[cols2]
B=test[cols2]
A = pd.get_dummies(A)
B = pd.get_dummies(B)

train_x=A.values
train_y=train['SalePrice'].values

test_x=B.values

my_imputer = Imputer()
test_x = my_imputer.fit_transform(test_x)

xgb_model = XGBRegressor(n_estimators=1000,learning_rate = 0.01, early_stopping_rounds =5,n_jobs=4)
xgb_model.fit(train_x, train_y, verbose=False)
predictions = xgb_model.predict(test_x)
pre = np.round(predictions)
subm = pd.DataFrame({'Id':test.Id,'SalePrice':pre})
subm.to_csv('mysubmission.csv', index = False)



