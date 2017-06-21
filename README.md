# Enron-Email-Classification
This project involves a preliminary text process, feature extraction and training the classifiers to distinguish spam or non-spam emails. 

## Data
 
The Raw data we used is from [Enron Corpus](https://enrondata.readthedocs.io/en/latest/data/edo-enron-email-pst-dataset/), which consists of 5172 training emails and 5857 testing emails in .txt format. Out of the 5172 training emails there are 1500 spam emails and 3672 ham emails. We are going to train the classification model with the training emails and to classify the testing set. Download the folder ```  ``` for the data.

## Python Script

The language used throughout will be [Python](https://www.python.org/), a general purpose language helpful in all parts of the pipeline: data preprocessing, model training and evaluation. While Python is by no means the only choice, it offers a unique combination of flexibility, ease of development and performance. Its vast, open source ecosystem also avoids the lock-in (and associated bitrot) of any single specific framework or library.

## Text Preprocessing Example

### Load Data

Load the text files and store them in lists.

```python
NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = './'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

def load_text_file(filenames):
    text_list = []
    for filename in filenames:
        with codecs.open(filename, "r", "utf-8", errors = 'ignore') as f:
            text = f.read().replace('\r\n', ' ')
            text_list.append(text)
    return text_list

spam_filenames = glob.glob( BASE_DIR + SPAM_DIR + '*.txt')
spam_list = load_text_file(spam_filenames)

ham_filenames = glob.glob( BASE_DIR + HAM_DIR + '*.txt')
ham_list = load_text_file(ham_filenames)

test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_list = load_text_file(test_filenames)

train_list = ham_list + spam_list
```
### Tokenizing
Here are some examples of tokenizing functions:

#### Tokenize on spaces
```python
train_list[1].split(' ')
```
#### Scikit-learn
```python
# note that CountVectorizer discards "words" that contain only one character, such as "s"
# CountVectorizer also transforms all words into lowercase
from sklearn.feature_extraction.text import CountVectorizer
CountVectorizer().build_tokenizer()(train_list[1])
```
#### NLTK
```python
# nltk word_tokenize uses the TreebankWordTokenizer and needs to be given a single sentence at a time.
from nltk.tokenize import word_tokenize
word_tokenize(train_list[1])
```
If you get an error here ```LookupError```, that means you need to download the NLTK Corpus:
```python
import nltk
nltk.download()
```

#### Maketrans(python 3 only)
```python
# see python documentation for string.translate
# string.punctuation is simply a list of punctuation
import string
table = str.maketrans({ch: None for ch in string.punctuation})
[s.translate(table) for s in train_list[1].split(' ') if s != '']
```
### Symbol removal
Sometimes we want to remove the meaningless symbols and numbers.

```python
from sklearn.feature_extraction.text import CountVectorizer
import re
trainlist1 = re.sub('[^A-Za-z]+', ' ', train_list[1])
tokenized = CountVectorizer().build_tokenizer()(trainlist1)
```

### Stemming
Stemming process reduces the number of unique vocabulary items that need to be tracked, it speeds up a variety of computational operations. NLTK offers stemming for a variety of languages in the [nltk.stem package](http://www.nltk.org/api/nltk.stem.html). The following code illustrates the use of the popular Porter stemmer:

```python
from nltk.stem.porter import *

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

stemmer = PorterStemmer()
stemmed = stem_tokens(tokenized, stemmer)
```

### Remove Stopwords

#### Download NLTK Corpus
```python
import nltk
nltk.download()
```
#### Remove
```python
from nltk.corpus import stopwords
filtered_words = [word for word in stemmed if word not in stopwords.words('english')]
```
## Email Classification

## Import packages
```python
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
```

## Feature extraction with TfidfVectorizer
Another option we can use is hhe function ```TfidfVectorizer```. It converts a collection of raw documents to a matrix of TF-IDF features, which is equivalent to CountVectorizer followed by TfidfTransformer.
Read more in the [User Guide](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction).

```python
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
all_filenames = spam_filenames + ham_filenames 

train_matrix = vectorizer.fit_transform(all_filenames)
test_matrix = vectorizer.transform(test_filenames)
X = train_matrix
Y = [1]*len(spam_filenames) + [0]*len(ham_filenames)

# Save as .mat 
file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_matrix
scipy.io.savemat('email_data.mat', file_dict)
```
## Cross-validation

### Support Vector Machine (Cross-validation)

```python
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
```
### Random Forest (Cross-validation)

```python
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
```
### Logistic Regression (Cross-validation)

```python
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
```
### Cross-validation code for other methods

```python
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
```
## Classification

### Support Vector Machine

```python
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
```
### Extreme Gradient Boost

```python
def xgboost_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    
    clf = XGBClassifier( max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8).fit(train_X, train_y)
    predicted = clf.predict(test_X)
    
    return predicted

submit_xgm = xgboost_submission(data)
```

### RandomForest

```python
def randomforest_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    
    clf = RandomForestClassifier(n_estimators=100).fit(train_X, train_y)
    predicted = clf.predict(test_X)
    
    return predicted

submit_rf = randomforest_submission(data)
```

### Logistic Regression

```python
def logreg_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    
    clf = LogisticRegression(C=1).fit(train_X, train_y)
    predicted = clf.predict(test_X)
    
    return predicted
    
submit_log = logreg_submission(data)
```
### Quadratic Discriminant Analysis

```python
def qda_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    
    clf = QuadraticDiscriminantAnalysis().fit(train_X, train_y)
    predicted = clf.predict(test_X)
    
    return predicted

submit_qda = qda_submission(data)
```

### Linear Discriminant Analysis

```python
def lda_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    
    clf = LinearDiscriminantAnalysis().fit(train_X, train_y)
    predicted = clf.predict(test_X)
    
    return predicted

submit_lda = lda_submission(data)
```

### Adaboost

```python
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
```
### Combine Results
```python
from scipy import stats
submit = [submit_log[i]+submit_rf[i]+submit_xgm[i]+submit_svm[i]+submit_ada[i] for i in xrange(len(submit_svm))]
submit = np.asarray(submit)

submit[np.where(submit==1)] = 0
submit[np.where(submit==2)] = 0
submit[np.where(submit==3)] = submit_log[np.where(a==3)]
submit[submit > 3] = 1
```
## Save as .csv file
```python
df = pd.DataFrame(submit)
df.index += 1
df['Id'] = df.index
df.columns = ['Category', 'Id']
#df = df['id'] + df['category']
df.to_csv("submit_new.csv",header=True,columns=['id','category'],index = False)
```
