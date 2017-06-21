
# coding: utf-8

# # Spam classification

# ## Import libraries

# In[5]:

import numpy as np
import pandas as pd
import time
import collections
import re
import random
import scipy.io
import glob

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from nltk import PorterStemmer


# ## Vectorizer

# In[8]:

NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = './'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

vectorizer = TfidfVectorizer(input='filename',lowercase=True, stop_words="english",
                             encoding='latin-1',min_df=8) 

spam_filenames = glob.glob( BASE_DIR + SPAM_DIR + '*.txt')
ham_filenames = glob.glob( BASE_DIR + HAM_DIR + '*.txt')
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
all_filenames = spam_filenames + ham_filenames # including test_filenames


train_matrix = vectorizer.fit_transform(all_filenames)
test_matrix = vectorizer.transform(test_filenames)
X = train_matrix
Y = [1]*len(spam_filenames) + [0]*len(ham_filenames)


#all_matrix = vectorizer.fit_transform(all_filenames)
#X = all_matrix[:NUM_TRAINING_EXAMPLES,:]
#Y = [1]*len(spam_filenames) + [0]*len(ham_filenames)
#test_matrix = all_matrix[NUM_TRAINING_EXAMPLES:,:]

# Save as .mat 
file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_matrix
scipy.io.savemat('email_data.mat', file_dict)


# In[9]:

print X.shape


# ### # of features

# In[15]:

len(vectorizer.vocabulary_)


# ### feature rank

# In[ ]:

occ = np.asarray(all_matrix.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences': occ})
counts_df.sort_values(by='occurrences', ascending=False).head(50)


# ##  Cross Validation

# In[ ]:

### SVM

def SVM_CV(data, n_samples, folds, C_hyperparam):
    #print "Problem 4: Spam Classification Cross Validation"
    
    X_raw = data['training_data'].toarray()
    y_raw = data['training_labels'].reshape(X_raw.shape[0],1)
    n = X_raw.shape[0]
    indices = np.arange(n)
    random.shuffle(indices)
    X_raw = X_raw[indices]
    y_raw = y_raw[indices]
    
    X = X_raw[:n_samples,:]
    y = y_raw[:n_samples]
    fold_size = len(y)/folds
    
    for c in C_hyperparam:
        print "{:<6}{:<6}{:<8}{:<6}".format('C', 'Fold', 'Error', 'Time')
        kfold_scores = []
        for i, x in enumerate(range(0, n_samples, fold_size)):
            start_time = time.time()
            all_idx  = set(range(0, n_samples))
            test_idx = set(range(x, x + fold_size))
            train_idx = sorted(list(all_idx - test_idx))
            test_idx  = sorted(list(test_idx))            

            train_fold_X, train_fold_y = X[train_idx], y[train_idx]
            test_fold_X,  test_fold_y  = X[test_idx],  y[test_idx]
            
            clf = LinearSVC(C=c).fit(train_fold_X, train_fold_y)
            predicted = clf.predict(test_fold_X)
            score = 1-accuracy_score(test_fold_y, predicted)
            kfold_scores.append(score)
            end_time = time.time()
            print "{:<6}{:<6}{:<8}{:<6}".format(c, i, round(np.mean(score),4), round(end_time - start_time, 2))

        mean_score = round(np.mean(kfold_scores), 4)
        mean_error = round(1 - mean_score, 4)
        print "Mean error: {}\nMean accuracy:    {}\n".format(mean_score, mean_error)


data = scipy.io.loadmat('./email_data.mat')
SVM_CV(data, n_samples=5000, folds=10, C_hyperparam=[.01, .1, 1, 10, 100])


# In[ ]:

### RANDOMFOREST

def randomforest_CV(data, n_samples, folds, C_hyperparam):
    #print "Problem 4: Spam Classification Cross Validation"
    
    X_raw = data['training_data'].toarray()
    y_raw = data['training_labels'].reshape(X_raw.shape[0],1)
    n = X_raw.shape[0]
    indices = np.arange(n)
    random.shuffle(indices)
    X_raw = X_raw[indices]
    y_raw = y_raw[indices]
    
    X = X_raw[:n_samples,:]
    y = y_raw[:n_samples]
    fold_size = len(y)/folds
    
    for c in C_hyperparam:
        print "{:<6}{:<6}{:<8}{:<6}".format('C', 'Fold', 'Error', 'Time')
        kfold_scores = []
        for i, x in enumerate(range(0, n_samples, fold_size)):
            start_time = time.time()
            all_idx  = set(range(0, n_samples))
            test_idx = set(range(x, x + fold_size))
            train_idx = sorted(list(all_idx - test_idx))
            test_idx  = sorted(list(test_idx))            

            train_fold_X, train_fold_y = X[train_idx], y[train_idx]
            test_fold_X,  test_fold_y  = X[test_idx],  y[test_idx]
            
            


            # fit model no training data
            clf = RandomForestClassifier(n_estimators=c).fit(train_fold_X, train_fold_y)
            predicted = clf.predict(test_fold_X)
            score = 1-accuracy_score(test_fold_y, predicted)
            kfold_scores.append(score)
            end_time = time.time()
            print "{:<6}{:<6}{:<8}{:<6}".format(c, i, round(np.mean(score),4), round(end_time - start_time, 2))

        mean_score = round(np.mean(kfold_scores), 4)
        mean_error = round(1 - mean_score, 4)
        print "Mean error: {}\nMean accuracy:    {}\n".format(mean_score, mean_error)


data = scipy.io.loadmat('./email_data.mat')
randomforest_CV(data, n_samples=5000, folds=5, C_hyperparam=[1, 10, 100, 1000])


# In[3]:

### logreg

def logistic_CV(data, n_samples, folds, C_hyperparam):
    #print "Problem 4: Spam Classification Cross Validation"
    
    X_raw = data['training_data'].toarray()
    y_raw = data['training_labels'].reshape(X_raw.shape[0],1)
    n = X_raw.shape[0]
    indices = np.arange(n)
    random.shuffle(indices)
    X_raw = X_raw[indices]
    y_raw = y_raw[indices]
    
    X = X_raw[:n_samples,:]
    y = y_raw[:n_samples]
    fold_size = len(y)/folds
    
    for c in C_hyperparam:
        print "{:<6}{:<6}{:<8}{:<6}".format('C', 'Fold', 'Error', 'Time')
        kfold_scores = []
        for i, x in enumerate(range(0, n_samples, fold_size)):
            start_time = time.time()
            all_idx  = set(range(0, n_samples))
            test_idx = set(range(x, x + fold_size))
            train_idx = sorted(list(all_idx - test_idx))
            test_idx  = sorted(list(test_idx))            

            train_fold_X, train_fold_y = X[train_idx], y[train_idx]
            test_fold_X,  test_fold_y  = X[test_idx],  y[test_idx]
            
            


            # fit model no training data
            clf =  LogisticRegression(C=c).fit(train_fold_X, train_fold_y)
            predicted = clf.predict(test_fold_X)
            score = 1-accuracy_score(test_fold_y, predicted)
            kfold_scores.append(score)
            end_time = time.time()
            print "{:<6}{:<6}{:<8}{:<6}".format(c, i, round(np.mean(score),4), round(end_time - start_time, 2))

        mean_score = round(np.mean(kfold_scores), 4)
        mean_error = round(1 - mean_score, 4)
        print "Mean error: {}\nMean accuracy:    {}\n".format(mean_score, mean_error)


data = scipy.io.loadmat('./email_data.mat')
logistic_CV(data, n_samples=5000, folds=5, C_hyperparam=[0.001, 0.01, 0.1, 1, 10, 100, 1000])


# ## GridsearchCV

# In[ ]:

data = scipy.io.loadmat('./email_data.mat')
n_samples = 3000

X_raw = data['training_data'].toarray()
y_raw = data['training_labels'].reshape(X_raw.shape[0],)
n = X_raw.shape[0]
indices = np.arange(n)
random.shuffle(indices)
X_raw = X_raw[indices]
y_raw = y_raw[indices]

X = X_raw[:n_samples,:]
y = y_raw[:n_samples]


param_test1 = {'max_depth':range(3,10,2),
               'min_child_weight':range(1,6,2)
              }

gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X,y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# ### Support Vector Machine

# In[3]:

def svm_submission(data, c):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    
    clf = LinearSVC(C=c).fit(train_X, train_y)
    predicted = clf.predict(test_X)
    return predicted

data = scipy.io.loadmat('./email_data.mat')
submit_svm = svm_submission(data, 0.1)


# ### Xgboost

# In[4]:

def xgboost_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    
    clf = XGBClassifier( max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8).fit(train_X, train_y)
    predicted = clf.predict(test_X)
    
    return predicted

submit_xgm = xgboost_submission(data)


# ### RandomForest

# In[5]:

def randomforest_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    
    clf = RandomForestClassifier(n_estimators=100).fit(train_X, train_y)
    predicted = clf.predict(test_X)
    
    return predicted

submit_rf = randomforest_submission(data)


# ### Logistic Regression

# In[13]:

def logreg_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    
    clf = LogisticRegression(C=1).fit(train_X, train_y)
    predicted = clf.predict(test_X)
    
    return predicted

submit_log = logreg_submission(data)


# ### Quadratic Discriminant Analysis

# In[ ]:

def qda_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    
    clf = QuadraticDiscriminantAnalysis().fit(train_X, train_y)
    predicted = clf.predict(test_X)
    
    return predicted

submit_qda = qda_submission(data)


# ### Linear Discriminant Analysis

# In[ ]:

def lda_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    
    clf = LinearDiscriminantAnalysis().fit(train_X, train_y)
    predicted = clf.predict(test_X)
    
    return predicted

submit_lda = lda_submission(data)


# ### adaboost

# In[ ]:

def adaboost_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    dt = DecisionTreeClassifier() 
    clf = AdaBoostClassifier(n_estimators=50, base_estimator=dt,learning_rate=1).fit(train_X, train_y)
    predicted = clf.predict(test_X)
    
    return predicted

submit_ada = adaboost_submission(data)


# ### Combine Results

# In[11]:

from scipy import stats
submit = [submit_log[i]+submit_rf[i]+submit_xgm[i]+submit_svm[i]+submit_ada[i] for i in xrange(len(submit_svm))]
submit = np.asarray(submit)

submit[np.where(submit==1)] = 0
submit[np.where(submit==2)] = 0
submit[np.where(submit==3)] = submit_log[np.where(a==3)]
submit[submit > 3] = 1


# ### Save as .csv

# In[ ]:

df = pd.DataFrame(submit)
df.index += 1
df['Id'] = df.index
df.columns = ['Category', 'Id']
#df = df['id'] + df['category']
df.to_csv("submit_new.csv",header=True,columns=['id','category'],index = False)





