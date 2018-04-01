import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

data = pd.read_csv('creditcard.csv', index_col='Time')
data.head()
data.describe()

out = 'Class'

import write_excel

reload(write_excel)
d_ob = write_excel.DataExplorationSheet(data_row=data.loc[4])
d_ob.save()

# Understanding data
plt.scatter(data[data['Class'] == 0]['V1'], data[data['Class'] == 0]['V2'],
            c='Green', label='Not Fraud')
plt.scatter(data[data['Class'] == 1]['V1'], data[data['Class'] == 1]['V2'],
            c='Red', label='Fraud')
plt.legend(loc='best')
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('Data Visualization')
# A check for missing data
print(data[data.isnull()].count())  # no missing data

# Separate Dependent and Independent variable
X = data.copy()
y = X.pop('Class')

# Split the data in training and testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.naive_bayes import GaussianNB

model_gaussian = GaussianNB()
model_gaussian.fit(X_train, y_train)
y_pred = model_gaussian.predict(X_test)

# Visualize the right and wrong predictions
comparision = y_pred == y_test
correct_pred = comparision[comparision == True]
incorrect_pred = comparision[comparision == False]
fig, ax = plt.subplots(1, 1)
ax.scatter(correct_pred.index, correct_pred, c='Green', s=3,
           label='Correct Predictions: ' + str(len(correct_pred)))
ax.scatter(incorrect_pred.index, incorrect_pred, s=3, c='Red',
           label='Incorrect Predictions' + str(incorrect_pred.count))
ax.legend(loc='best')
fig.savefig('Result_visualization.png')

# compare the predictions with actual values
from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(y_test, y_pred)

# Let's check the accuracy with cross validation
from sklearn.model_selection import cross_val_score

accuracy = np.average(cross_val_score(model_gaussian, X_train, y_train, cv=7))
print(accuracy)  # .9775 percent

# This simple model gave us 97% accuracy. Let's try Support Vector Classifier on it
from sklearn.svm import SVC

# This model takes a imp parameter C. Let's find it's value using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [1, 2, 4]}
grid = GridSearchCV(SVC(), param_grid=param_grid)
grid.fit(X_train, y_train)

print(grid.best_params_)  # {'C': 4}

model_svc = grid.best_estimator_
model_svc.fit(X_train, y_train)
y_pred = model_svc.predict(X_test)

cmat_svc = confusion_matrix(y_test, y_pred)

# See accuracy with cross validation
accuracy_svc = np.average(cross_val_score(model_gaussian, X_train, y_train, cv=7))
print(accuracy_svc)
