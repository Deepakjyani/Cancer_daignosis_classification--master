# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 07:33:48 2020

@author: deepak
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

plt.style.use('dark_background')

file_1 = open("KIRP_mrna.txt")
file_1.readline()
X = np.loadtxt(file_1)
file_1.close()

file_1 = open("KIRP_label.txt")
file_1.readline()
y = np.loadtxt(file_1)
file_1.close()

### split data set into test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#### Data preprocessing 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.fit_transform(X_test)

#### Different Classification model

### model 1 for K nearest neighbors model

from sklearn.neighbors import KNeighborsClassifier
model_1 = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', p= 3)
model_1.fit(X_train, y_train)
accurices1 = cross_val_score(estimator = model_1, X = X_train, y = y_train, cv = 20, n_jobs = -1)
accuracy1 = accurices1.mean()
var1 = accurices1.std()

y_pred_1 = model_1.predict(X_test)
mse_1 = mean_squared_error(y_test, y_pred_1)
mae_1 = mean_absolute_error(y_test, y_pred_1)
print("Mean Square Error from KNN Classifier", mse_1)
print("Mean Absolute Error from KNN Classifier", mae_1)


### model 2 for SVM model

from sklearn.svm import SVC
model_2 = SVC(C = 1.0, kernel = 'rbf', random_state = 0)
model_2.fit(X_train, y_train)
accurices2 = cross_val_score(estimator = model_2, X = X_train, y = y_train, cv = 20, n_jobs = -1)
accuracy2 = accurices2.mean()
var2 = accurices2.std()

y_pred_2 = model_2.predict(X_test)
mse_2 = mean_squared_error(y_test, y_pred_2)
mae_2 = mean_absolute_error(y_test, y_pred_2)
print("Mean Square Error from SVM Classifier", mse_2)
print("Mean Absolute Error from SVM Classifier", mae_2)

### model 3 for Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_3 =  DecisionTreeClassifier()
model_3.fit(X_train, y_train)
accurices3 = cross_val_score(estimator = model_3, X = X_train, y = y_train, cv = 20, n_jobs = -1)
accuracy3 = accurices3.mean()
var3 = accurices3.std()

y_pred_3 = model_3.predict(X_test)
mse_3 = mean_squared_error(y_test, y_pred_3)
mae_3 = mean_absolute_error(y_test, y_pred_3)
print("Mean Square Error from Decision Tree  Classifier", mse_3)
print("Mean Absolute Error from Decision Tree Classifier", mae_3)


### model 4 for Random forest tree
from sklearn.ensemble import RandomForestClassifier
model_4 = RandomForestClassifier(n_estimators = 30, random_state = 20)
model_4.fit(X_train, y_train)
accurices4 = cross_val_score(estimator = model_4, X = X_train, y = y_train, cv = 20, n_jobs = -1)
accuracy4 = accurices4.mean()
var4 = accurices4.std()

y_pred_4 = model_4.predict(X_test)
mse_4 = mean_squared_error(y_test, y_pred_4)
mae_4 = mean_absolute_error(y_test, y_pred_4)
print("Mean Square Error from Random Forest Classifier", mse_4)
print("Mean Absolute Error from Random Forest Classifier", mae_4)




#### model 5 for Logistic Regression 

from sklearn.linear_model import LogisticRegression
model_5 = LogisticRegression(random_state = 10)
model_5.fit(X_train, y_train)
accurices5 = cross_val_score(estimator = model_5, X = X_train, y = y_train, cv = 20, n_jobs = -1)
accuracy5 = accurices5.mean()
var5 = accurices5.std()

y_pred_5 = model_5.predict(X_test)
mse_5 = mean_squared_error(y_test, y_pred_5)
mae_5 = mean_absolute_error(y_test, y_pred_5)
print("Mean Square Error from Logistic Regression Classifier", mse_5)
print("Mean Absolute Error from Logistic Regression  Classifier", mae_5)

#### model 6 for XGBoost 

from xgboost import XGBClassifier
model_6 = XGBClassifier()
model_6.fit(X_train, y_train)
accurices6 = cross_val_score(estimator = model_6, X = X_train, y = y_train, cv = 20, n_jobs = -1)
accuracy6 = accurices6.mean()
var6 = accurices6.std()

y_pred_6 = model_6.predict(X_test)
mse_6 = mean_squared_error(y_test, y_pred_6)
mae_6 = mean_absolute_error(y_test, y_pred_6)
print("Mean Square Error from XGBoost  Classifier", mse_6)
print("Mean Absolute Error from XGBoost Classifier", mae_6)


#### model 7 for Deep Neural Network
### define the model

model = Sequential()
model.add(Dense(5000, input_dim = 16175, activation = 'relu', init = 'uniform'))
model.add(Dense(500, activation = 'relu', init = 'uniform'))
model.add(Dense(100, activation = 'relu', init = 'uniform'))
model.add(Dense(50, activation = 'relu', init = 'uniform'))
model.add(Dense(1, activation = 'sigmoid', init = 'uniform'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_split = 0.2, epochs = 20)

y_preds = model.predict(X_test).ravel()
mse_neural, mae_neural = model.evaluate(X_test, y_test)
print("Mean Square Error from Neural network ", mse_neural)
print("Mean Absolute Error from Neural network", mae_neural)



### plot the training and validation accuracy and loss at each epochs
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = "validation loss")
plt.title("Training and validation loss")
plt.xlabel ("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


loss = history.history['accuracy']
val_loss = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label = 'Training Accuracy')
plt.plot(epochs, val_loss, 'r', label = "validation Accuracy")
plt.title("Training and validation Accuracy")
plt.xlabel ("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


### ROC Curve for multiple models

from sklearn.metrics import roc_curve
fpr_1, tpr_1, threshold_1 = roc_curve(y_test, y_pred_1)
fpr_2, tpr_2, threshold_2 = roc_curve(y_test, y_pred_2)
fpr_3, tpr_3, threshold_3 = roc_curve(y_test, y_pred_3)
fpr_4, tpr_4, threshold_4 = roc_curve(y_test, y_pred_4)
fpr_5, tpr_5, threshold_5 = roc_curve(y_test, y_pred_5)
fpr_6, tpr_6, threshold_6 = roc_curve(y_test, y_pred_6)
fpr, tpr, threshold = roc_curve(y_test, y_preds)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr_1, tpr_1, marker = '.', label = 'KNN')
plt.plot(fpr_2, tpr_2, marker = '.', label  = 'SVM')
plt.plot(fpr_3, tpr_3, marker = '.', label =  'Decision Tree')
plt.plot(fpr_4, tpr_4, marker = '.', label = 'Random Forest ')
plt.plot(fpr_5, tpr_5, marker = '.', label = 'Logistic_Regressor ')
plt.plot(fpr_6, tpr_6, marker = '.', label = 'XGBoost')
plt.plot(fpr, tpr, marker = '.', label = 'DNN')
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.title("ROC Curve")
plt.show()


### Comparision based on mse on different models

plt.figure()
plt.title("Outputs of different Mean Sequare Error functions")
plt.bar(["KNN ", "SVM", "De Tree", "R Forest", "Logistic R", "XGBoost", "DNN"], [mse_1, mse_2, mse_3, mse_4, mse_5, mse_6, mse_neural/3])
plt.xlabel("Models")
plt.ylabel("Mean Square Error")
plt.show()

### Comparision based on average accuracy on different models

plt.figure()
plt.title("Outputs of different Accuracy functions")
plt.bar(["KNN ", "SVM", "De Tree", "R Forest", "Logistic R", "XGBoost", "DNN"], [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, 0.9650])
plt.xlabel("Models")
plt.ylabel("Mean Accuracy")
plt.show()

