# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 18:08:51 2020

@author: deepak
"""

import numpy as np
from matplotlib import pyplot as plt 

file_1=open("KIRC_mirna.txt")
file_1.readline()
X=np.loadtxt(file_1)
file_1.close()

file_4=open("KIRC_label.txt")
file_4.readline()
y=np.loadtxt(file_4)
file_4.close()

### train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

### ss
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)


###dimenssion reduction

from sklearn.decomposition import PCA
pca = PCA(n_components = 15, random_state = 43)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)


### classification methods

### 1. kneighborsclassification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 9, weights = 'uniform')
knn.fit(X_train, y_train)
accurices1 = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 20, n_jobs = -1)

### 2. svm 
from sklearn.svm import SVC
svc = SVC(kernel = 'linear', probability = True, C = 0.01 )
svc.fit(X_train, y_train)

accurices2 = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = 20, n_jobs = -1)

### 3. Naive bayes

from sklearn.naive_bayes import GaussianNB
naive = GaussianNB(var_smoothing = 1e-09)
naive.fit(X_train, y_train)

accurices3 = cross_val_score(estimator = naive, X = X_train, y = y_train, cv = 20, n_jobs = -1)


### 4. Decision Tree

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree.fit(X_train, y_train)

accurices4 = cross_val_score(estimator = tree, X = X_train, y = y_train, cv = 20, n_jobs = -1)


### 5. Random Forest

from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state = 0)
randomforest.fit(X_train, y_train)

accurices5 = cross_val_score(estimator = randomforest, X = X_train, y = y_train, cv = 20, n_jobs = -1)



### 6. XGBoost

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

accurices6 = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 20, n_jobs = -1)


#### 7. Deep Neural Network

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import metrics

classifier = Sequential()
classifier.add(Dense(input_dim = 15, init = 'uniform', activation = 'relu', output_dim = 20))
#classifier.add(Dropout(0.5))
classifier.add(Dense(init = 'uniform', activation = 'relu', output_dim = 25))
#classifier.add(Dropout(0.4))
classifier.add(Dense(init = 'uniform', activation = 'relu', output_dim = 21))
#classifier.add(Dropout(0.2))
classifier.add(Dense(init = 'uniform', activation = 'relu', output_dim = 15))
classifier.add(Dense(init = 'uniform', activation = 'relu', output_dim = 10))
classifier.add(Dense(init = 'uniform', activation = 'relu', output_dim = 7))
classifier.add(Dense(init = 'uniform', activation = 'relu', output_dim = 3))
classifier.add(Dense(init = 'uniform', activation = 'sigmoid', output_dim = 1))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit( x = X_train, y = y_train, batch_size = 300, epochs = 100)
y_pred7 = classifier.predict_proba(X_test)
y_pred7 = (y_pred7 > 0.35)
y_proba7 = classifier.predict_proba(X_test)




from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(input_dim = 15, init = 'uniform', activation = 'relu', output_dim = 20))
    classifier.add(Dense(init = 'uniform', activation = 'relu', output_dim = 25))
    #classifier.add(Dropout(0.4))
    classifier.add(Dense(init = 'uniform', activation = 'relu', output_dim = 21))
    #classifier.add(Dropout(0.2))
    classifier.add(Dense(init = 'uniform', activation = 'relu', output_dim = 15))
    classifier.add(Dense(init = 'uniform', activation = 'relu', output_dim = 10))
    classifier.add(Dense(init = 'uniform', activation = 'relu', output_dim = 7))
    classifier.add(Dense(init = 'uniform', activation = 'relu', output_dim = 3))
    classifier.add(Dense(init = 'uniform', activation = 'sigmoid', output_dim = 1))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 300, epochs = 100)
accuracies7 = cross_val_score( estimator = classifier, X = X_train, y = y_train, cv = 20)



### Best model for given dataset

accuracy = [accurices1.mean(), accurices2.mean(), accurices3.mean(),accurices4.mean(),accurices5.mean(), accurices6.mean(),accuracies7.mean()]
varience = [accurices1.std(), accurices2.std(), accurices3.std(), accurices4.std(), accurices5.std(), accurices6.std(), accuracies7.std()]
Best_Accuracy = (max(accuracy), min(varience))
worst_Accuracy = (min(accuracy), max(varience))


from sklearn.metrics import auc, roc_curve


y_proba = [knn.predict_proba(X_test), svc.predict_proba(X_test), 
			naive.predict_proba(X_test), tree.predict_proba(X_test),
			randomforest.predict_proba(X_test), xgb.predict_proba(X_test),
            ]
y_proba.append(y_proba7)

y_pred = [knn.predict(X_test), svc.predict(X_test),
          naive.predict(X_test), tree.predict(X_test),
          randomforest.predict(X_test), xgb.predict(X_test)]
y_pred.append(y_pred7)


cmatrix = []
cmatrix.append(confusion_matrix(y_test, y_pred[0]))
cmatrix.append(confusion_matrix(y_test, y_pred[1]))
cmatrix.append(confusion_matrix(y_test, y_pred[2]))
cmatrix.append(confusion_matrix(y_test, y_pred[3]))
cmatrix.append(confusion_matrix(y_test, y_pred[4]))
cmatrix.append(confusion_matrix(y_test, y_pred[5]))
cmatrix.append(confusion_matrix(y_test, y_pred[6]))

acc = [metrics.roc_auc_score(y_test, y_proba[0][:, 1]),metrics.roc_auc_score(y_test, y_proba[1][:, 1]),
       metrics.roc_auc_score(y_test, y_proba[2][:, 1]), metrics.roc_auc_score(y_test, y_proba[3][:, 1]),
       metrics.roc_auc_score(y_test, y_proba[4][:, 1]), metrics.roc_auc_score(y_test, y_proba[5][:, 1]),
       metrics.roc_auc_score(y_test, y_proba7)]



precision1, recall1, _thresholds1 = metrics.precision_recall_curve(y_test, y_proba[0][:, 1])
precision2, recall2, _thresholds2 = metrics.precision_recall_curve(y_test, y_proba[1][:, 1])
precision3, recall3, _thresholds3 = metrics.precision_recall_curve(y_test, y_proba[2][:, 1])
precision4, recall4, _thresholds4 = metrics.precision_recall_curve(y_test, y_proba[3][:, 1])
precision5, recall5, _thresholds5 = metrics.precision_recall_curve(y_test, y_proba[4][:, 1])
precision6, recall6, _thresholds6 = metrics.precision_recall_curve(y_test, y_proba[5][:, 1])
precision7, recall7, _thresholds7 = metrics.precision_recall_curve(y_test, y_proba[6])

pr_auc = [metrics.auc(recall1, precision1), metrics.auc(recall2, precision2),
          metrics.auc(recall3, precision3), metrics.auc(recall4, precision4),
          metrics.auc(recall5, precision5), metrics.auc(recall6, precision6),
          metrics.auc(recall7, precision7)]

mcc = [matthews_corrcoef(y_test,y_pred[0]), matthews_corrcoef(y_test,y_pred[1]),
       matthews_corrcoef(y_test,y_pred[2]), matthews_corrcoef(y_test,y_pred[3]),
       matthews_corrcoef(y_test,y_pred[4]), matthews_corrcoef(y_test,y_pred[5]),
       matthews_corrcoef(y_test,y_pred[6])]


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')

#plt.plot(recall1, precision1, color = 'pink', label = 'KNN')
#plt.plot(recall2, precision2, color = 'black', label = 'SVM')
#plt.plot(recall3, precision3, color = 'red', label = 'Naive')
#plt.plot(recall4, precision4, color = 'cyan', label = 'Decision Tree')
#plt.plot(recall5, precision5, color = 'green', label = 'Random Forest')
#plt.plot(recall6, precision6, color = 'magenta', label = 'XGBoost')
#plt.plot(recall7, precision7, color = 'blue', label = 'DNN')


precision = [precision1.mean(), precision2.mean(), precision3.mean(),
             precision4.mean(), precision5.mean(),
             precision6.mean(), precision7.mean()]


recall = [recall1.mean(), recall2.mean(), recall3.mean(),
             recall4.mean(), recall5.mean(),
             recall6.mean(), recall7.mean()]

plt.plot(acc, label = 'acc')
plt.plot(accuracy)
plt.plot(pr_auc)
plt.plot(mcc)
plt.show()