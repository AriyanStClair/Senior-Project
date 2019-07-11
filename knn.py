#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy, sklearn, urllib, librosa, IPython.display as ipd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVC 
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix


# In[2]:


# load features.csv
df = pd.read_csv('features.csv')
#df.head()


# In[3]:


# csv file without filename

features = df.drop(columns=["filename"])
#features.head()


# In[4]:


#check number of rows and columns in dataset
features.shape


# In[5]:


#create a dataframe with all training data except the target column
X = features.drop(columns=["genre"])
#check that the target variable has been removed
#X.head()


# In[6]:


#separate target values
y = df["genre"].values


# # KNN

# In[7]:


from sklearn.model_selection import GridSearchCV
#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)
#check top performing n_neighbors value
knn_gscv.best_params_


# In[8]:


#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_


# In[9]:


# cross validation set mean

from sklearn.model_selection import cross_val_score
import numpy as np
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=19)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, y, cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean: {}'.format(np.mean(cv_scores)))


# In[10]:


#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 19)
# Fit the classifier to the data
knn.fit(X_train,y_train)

#make predictions on the test data
y_pred = knn.predict(X_test)

#check accuracy of our model on the test data
print('Accuracy: {0:.3f} %'.format(knn.score(X_test, y_test) * 100))


# In[11]:


print(classification_report(y_test,y_pred))  


# In[12]:


# Confusion Matrix
import seaborn as sn
genres = df["genre"].unique()
cmx = confusion_matrix(y_test,y_pred)
df_cm = pd.DataFrame(cmx,genres,genres)
sn.set(font_scale = 1.4)
sn.heatmap(df_cm, annot = True, annot_kws = {"size": 16})
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

