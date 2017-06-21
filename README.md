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


