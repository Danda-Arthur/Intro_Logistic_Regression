from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# # Shows how many columns should be displayed of the data output
# pd.options.display.max_columns = 30

breast_cancer_dataset = load_breast_cancer()
breast_cancer = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
breast_cancer['result'] = breast_cancer_dataset['target']

# # Give a detailed description of the dataset
# print(breast_cancer_dataset.keys())
# print(breast_cancer_dataset['DESCR'])

# In order to know what the target set represents
# print(breast_cancer_dataset['target_names'])
# output is -> ['malignant' 'benign'] showing that a 0=>malignant and 1=>benign

# Bulding a Logistic Regression model
# Can also instantiate X and y without .values  (.values changes the Dataframe to a numpy array
X = breast_cancer.values
y = breast_cancer['result'].values
'''
model = LogisticRegression()
model.fit(X, y)
# If we try to fit using the normal we get a Convergence Warning meaning the model needs more time to find the optimal
# solution, This can be solved by using a different solver  '''

# '''
                # THIS IS THE ACTUAL CODE #
# model = LogisticRegression(solver='liblinear')
# model.fit(X, y)
# # print(model.predict([X[90]]))
# # print(model.score(X, y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=1)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Using metric functions in determining accuracy, precision, recall and F1 score
y_pred = model.predict(X_test)

# the import function statement of these metric functions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# each function takes two 1-dimensional numpy arrays: the true values and the predicted values of the target
print("accuracy: ", accuracy_score(y_test, y_pred))
print("precision: ", precision_score(y_test, y_pred))
print("recall: ", recall_score(y_test, y_pred))
print("f1_score: ", f1_score(y_test, y_pred))

# using scikit-learn we are able to get the four values in the confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
# '''

'''
                # USING ROC CURVE #
# The ROC curve is a graph of the specificity vs sensitivity i.e specificity is the % actual negatives correctly
# predicted and sensitivity being the % actual positives correctly predicted
# We have to use the train_split function for the ROC curve
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')

plt.show()  '''

'''
# In order for us to build and evaluate multiple models of the dataset, we split the dataset into multiple training and
# testing sets. We use k-fold cross validation in order to split the data. The k is the number of chunks we split our
# dataset into. However the models which are built are just for evaluation purposes, so that we can report the metric
# values. We don't actually need these models and want to build the best possible model. The purpose of this is model
# checking and not model building.
from sklearn.model_selection import KFold

new_X = X
new_y = y
kf = KFold(n_splits=5, shuffle=True)

splits = list(kf.split(new_X))
first_split = splits[0]
# print(first_split) -> result is (array([2, 3, 4, 5]), array([0, 1])), the first array is the indices for the training
# set and the second is the indices for the test test
train_indices, test_indices = first_split

X_train = new_X[train_indices]
X_test = new_X[test_indices]
y_train = new_y[train_indices]
y_test = new_y[test_indices]

print('X_train', X_train)
print('X_test', X_test)
print('y_train', y_train)
print('y_test', y_test)
'''
