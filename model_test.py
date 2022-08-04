import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from pickle import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.linear_model import LinearRegression

boston = load_boston()
df=pd.DataFrame(boston.data,columns=boston.feature_names)
df["Price"]=boston.target
df.head()

#logarithmic transform
for feature in df.columns:
    if 0 not in df[feature].unique():
        df[feature]=np.log(df[feature])

X = df.drop(['Price'], axis = 1)
y = pd.DataFrame(df['Price'])

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 43)

#scaling
scale=[feature for feature in X_train.columns if feature not in ['Price']]

scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
dump(scaler, open('scaler.pkl', 'wb'))

X_train=pd.DataFrame(X_train,columns=X.columns)
X_test=pd.DataFrame(X_test,columns=X.columns)

#Model Trainig

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
dump(lin_model, open('model.pkl', 'wb'))

print("Accuracy on the Train data is {} % for our Linear Regression model.".format(100*metrics.r2_score(y_train,lin_model.predict(X_train))))
print("Accuracy on the Test data is {} % for our Linear Regression model.".format(100*metrics.r2_score(y_test,lin_model.predict(X_test))))



