#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


get_ipython().system('pip install numpy pandas seaborn')


# In[2]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 


# In[4]:


get_ipython().system('pip install js')


# In[ ]:





# In[5]:


import requests
URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
response = requests.get(URL1)
print(response.text)


# In[6]:


import io


# In[7]:


import httpx
async with httpx.AsyncClient() as client:
    response = await client.get("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")
    text1 = io.BytesIO(response.content)
    response1 = await client.get("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv")
    text2 = io.BytesIO(response1.content)


# In[8]:


data = pd.read_csv(text1)
X = pd.read_csv(text2)


# In[9]:


data.head()


# In[10]:


X.head(100)


# In[11]:


Y=data['Class'].to_numpy()


# In[12]:


type(Y)


# In[13]:


from sklearn.preprocessing import StandardScaler
preprocessing = StandardScaler()
X_standardized = pd.DataFrame(preprocessing.fit_transform(X), columns=X.columns)


# In[14]:


X_standardized.head(100)


# In[15]:


X = X_standardized


# In[16]:


X.head(10)


# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[72]:



X_test.shape


# In[19]:


parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}


# In[20]:


parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()


# In[21]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(lr, parameters, cv=3)
grid_search.fit(X_train, Y_train)
best_params_= grid_search.best_params_
best_score_ = grid_search.best_score_


print("tuned hpyerparameters :(best parameters) ",best_params_)
print("accuracy :",best_score_)


# In[22]:


grid_search = GridSearchCV(lr, parameters, cv=3)
grid_search.fit(X_test, Y_test)
best_params_t_= grid_search.best_params_
best_score_t = grid_search.best_score_


# In[23]:


print("tuned hpyerparameters :(best parameters) ",best_params_t_)
print("accuracy :",best_score_t)


# In[24]:


#best_lr = LogisticRegression(**parameters)

best_lr = LogisticRegression(**best_params_t_)


# In[25]:


best_lr.fit(X_train, Y_train)


# In[26]:


test_accuracy = best_lr.score(X_test, Y_test)


# In[27]:


print("Test Accuracy is :", test_accuracy)


# In[29]:


parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}


# In[30]:


svm = SVC()


# In[31]:


print(sum)


# In[38]:


svm_cv = GridSearchCV(estimator=svm, param_grid=parameters, cv=10, scoring='accuracy')


# In[39]:


svm_cv.fit(X_train, Y_train) 


# In[40]:


print("Best Hyperparameters:", svm_cv.best_params_)


# In[41]:


best_svm_model = svm_cv.best_estimator_


# In[36]:


accuracy_on_test_set = best_svm_model.score(X_test, Y_test)  # X_test and Y_test are your test data and labels
print("Accuracy on Test Set:", accuracy_on_test_set)


# In[42]:


yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# In[43]:


tree = DecisionTreeClassifier()


# In[44]:


parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}


# In[46]:


tree_cv = GridSearchCV(estimator=tree, param_grid=parameters, cv=10, scoring='accuracy')
    


# In[47]:


tree_cv.fit(X_train, Y_train)


# In[49]:


print("Best Hyperparameters:", tree_cv.best_params_)


# In[50]:


best_tree_model = tree_cv.best_estimator_


# In[51]:


ccuracy_on_test_set = best_tree_model.score(X_test, Y_test)  # Replace X_test and Y_test with your actual test data and labels
print("Accuracy on Test Set:", accuracy_on_test_set)


# In[52]:


tree_cv.fit(X_test, Y_test)


# In[53]:


print("Best Hyperparameters:", tree_cv.best_params_)


# In[54]:


yhat = tree_cv.predict(X_test)


# In[68]:


Accuracy_on_test_set_Tree = tree_cv.score(X_test, Y_test)


# In[55]:


plot_confusion_matrix(Y_test,yhat)


# In[56]:


KNN = KNeighborsClassifier()


# In[57]:


parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}


# In[59]:


KNN_CV= GridSearchCV(estimator=KNN, param_grid=parameters, cv=10, scoring='accuracy')


# In[61]:


KNN_CV.fit(X_train, Y_train)


# In[62]:


print("Best Hyperparameters:", KNN_CV.best_params_)


# In[65]:


y_pred = KNN_CV.predict(X_test)


# In[66]:


Accuracy_on_test_set = KNN_CV.score(X_test, Y_test)


# In[67]:


print("Accuracy on Test Set:", accuracy_on_test_set)


# In[69]:


print("Accuracy on Test Set:", Accuracy_on_test_set_Tree)


# In[70]:


print("Test Accuracy is :", test_accuracy)


# In[ ]:


best_lr

