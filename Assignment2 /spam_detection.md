
*<h1>TEXT CLASSIFICATION: SPAM DETECTION</h1>*
**<h4>Author: Kris Yu<h4>**

*<h3>Introduction</h3>*
Email detection is one of the most importand applications of Natural Language Processing. The common ways are Hand-coded rules which is quite complicated in processing with low recall rate and supervised learning which includes 
Naïve Bayes、Logistic regression、Support-vector machines、k-Nearest Neighbors, etc. Here I will use the Naïve Bayes Classifier since it is more straightforward and simpler than other algorithms.

*<h3>Algorithm Details</h3>*
Naïve Bayes is based on bayes rule and bag of words analysis with a very important assumption: tokens appeard in the text are disorderly and independant with each other. The Naïve Bayes Classifier can be explained like this:
$$P(c|d) = \frac{P(d|c)P(c)}{P(d)}$$ where d represents document and c represents classes. The class c of the given document d is:
$$
\begin{split}
C &= argmax\ P(c|d) \\
&=argmax\ \frac{P(d|c)P(c)}{P(d)}\\
&=argmax\ P(d|c)P(c)\\
&=argmax\ P(t_1,t_2,...,t_n|c)P(c)\\
&=argmax\ P(t_1|c)P(t_2|c),...,P(t_n|c)P(c) \\
&=argmax\ logP(t_1|c)+logP(t_2|c)+,...,+logP(t_n|c)+logP(c)
\end{split}
$$
where $t_1,...,t_n$ are tokens in the given text. $P(c)$ is the prior probability, $P(t_1,t_2,...,t_n|c)$ is the conditional probability, namely likelihood and $P(c|d)$ is the posterior probability.

*<h3>Training Phase</h3>*

Our training data are emails in the SPAM_trainig folder with the file name like 'HAM.02281.txt' representing ham email and 'SPAM.02281.txt' representing spam email.


```python
import nltk
import pickle
import math
import re
from collections import Counter
```

* <h4>Load our training files and take a look at the contents</h4>

ham stores all true emails with the file name like HAM.0123.txt while spam stores all spam emails with the file name like SPAM.0234.txt


```python
ham = nltk.corpus.PlaintextCorpusReader('/Users/Aslan/winter 2017/Lin127/hw2,3/SPAM_training', 'HAM.*.txt')
spam = nltk.corpus.PlaintextCorpusReader('/Users/Aslan/winter 2017/Lin127/hw2,3/SPAM_training', 'SPAM.*.txt')
```

There are 13545 ham emails and 4912 spam emails in our training dataset.


```python
# number of files in each class
ham_count = len(ham.fileids())
spam_count = len(spam.fileids())
# total number of tokens in each class
ham_tokens = ham.words()
spam_tokens = spam.words()
print(ham_count, spam_count)
```

    13545 4912


* <h4>Build our text classifier</h4>


```python
def get_freq_df(words):
    '''Build a nltk frequency table for the token types in each class
    
    Args:
        words(nltk object): a list of tokens
    
    Returns:
        dist_words(nltk FreqDist): frequency table
    '''
    dist_words = nltk.FreqDist(words)
    return dist_words
```


```python
def solver(spam, ham, spam_count, ham_count):
    '''build the solver model
    Args: 
        spam(nltk object): a list of tokens in spam
        ham(nltk object): a list of tokens in ham
    Returns:
        a dict with two data frames and two counts for each class 
    '''
    df_spam = get_freq_df(spam)
    df_ham = get_freq_df(ham)

    return {'spam_fd': df_spam, 'ham_fd': df_ham, 
            'spam_count': spam_count, 'ham_count': ham_count}
```

* <h4>Save our model into spam.nb for time saving when reusing this code.</h4>


```python
solver_NB = solver(spam_tokens, ham_tokens, spam_count, ham_count)
with open('spam.nb', 'wb') as f:
    pickle.dump(solver_NB, f, protocol=pickle.HIGHEST_PROTOCOL)
```

*<h3>Testing Phase</h3>*

* <h4>we reload our classifier model from our hard disk</h4>


```python
with open('spam.nb', 'rb') as f:
    model = pickle.load(f)
```

log_prior_spam/ham represents the prior logrithm probability of spam/ham class.
we can see that there are, in total,  1284301 tokens in spam training data and 4903935 in ham data.


```python
log_prior_spam = math.log(model['spam_count']/(model['spam_count'] + model['ham_count']))
log_prior_ham = math.log(model['ham_count']/(model['spam_count'] + model['ham_count']))
spam_n = model['spam_fd'].N()
ham_n = model['ham_fd'].N()
print(spam_n, ham_n)
```

    1284301 4903935


* <h4>create the likelihood table with add-one-smoothing modification</h4>

The add-one-smoothing is one of the methods to fix the new word problem. When there are new words which can be found only from one of our training class(spam or ham) in the text files, the posterior probablity could be zero since $P(new|c) = 0$ for the class that does not include that new word. We don't want our classifier be disturbed by these new words and that is why we should implement this modification here.

To be more detailed:
$$P(token|c) = \frac{(number\ of\ the\ given\ token\ in\ class\ c) +1}{(token\ size\ of\ class\ c) + (vocabulary\ size)}$$ where the vocabulary size is the total number of token type in the union of spam and ham training set

However, add-one-smoothing can't fix the scenario where the new word doesn't show in any traning class. In that case, I just simply omit that word.


```python
# Build the vacabulary set on the union of spam and ham training set.
vocabulary = set(list(model['ham_fd'].keys()) + list(model['spam_fd'].keys()))
vocal_size = len(vocabulary)
vocal_size
```




    101357




```python
spam_log_likelihood = {}    # initialize the conditional probablity
ham_log_likelihood = {}
# add-one-smoothing
for token in vocabulary:
    spam_log_likelihood[token] = math.log((model['spam_fd'][token] + 1) / (spam_n + vocal_size))
    ham_log_likelihood[token] = math.log((model['ham_fd'][token] + 1) / (ham_n + vocal_size))
```


```python
def spam_detection(file, ham_log_likelihood, spam_log_likelihood, 
                   log_prior_spam, log_prior_ham, vocabulary):
    ''' classify the given file as a binary outcome(spam or ham)
    Args:
        file: file name(eg.'test.0123.txt')
        ham(spam)_log_likelihood: likelihood table after taking add-one-smoothing and logrithm
        log_prior_spam(ham): prior probability in log type
        vocabulary: token types in the union of spam and ham training dataset.
    Returns:
        print file name with 'SPAM' or 'HAM'
    '''       
    token_ls = dev.words(fileids=file)
    log_posterior_spam = log_prior_spam
    log_posterior_ham = log_prior_ham
    
    for token in token_ls:
        if not token in vocabulary:
            continue
        log_posterior_spam += spam_log_likelihood[token]
        log_posterior_ham += ham_log_likelihood[token]
    
    if log_posterior_spam > log_posterior_ham:
        print(file, 'SPAM')
    else:
        print(file, 'HAM')

```

* <h4>Run the classifier on our test data</h4>

I wrote the code above in a more decent way in a script file called nbtest.py and run the shell command: 

**python3 nbtest.py > SPAM_dev_predictions.txt**

Then we load the result into variable dev by nltk


```python
dev = nltk.corpus.PlaintextCorpusReader('/Users/Aslan/winter 2017/Lin127/hw2,3/', 'SPAM_dev_predictions.txt')
```

*<h3>Model Evaluation Phase</h3>*


```python
file_ls = dev.raw().strip().split('\n')
confusion = [re.search(r'(HAM|SPAM)\..*(HAM|SPAM)', i).group(1) + ' ' +
             re.search(r'(HAM|SPAM)\..*(HAM|SPAM)', i).group(2)
             for i in file_ls]
file_ls[:10]
```




    ['HAM.00039.txt HAM',
     'HAM.00045.txt HAM',
     'HAM.00064.txt HAM',
     'HAM.00091.txt HAM',
     'HAM.00098.txt HAM',
     'HAM.00137.txt HAM',
     'HAM.00143.txt HAM',
     'HAM.00146.txt HAM',
     'HAM.00153.txt HAM',
     'HAM.00205.txt HAM']




```python
Counter(confusion)
```




    Counter({'HAM HAM': 986, 'HAM SPAM': 14, 'SPAM HAM': 14, 'SPAM SPAM': 349})



Our test set has 1000 ham and 363 spam
Therefore: 

$recall(HAM) = \frac{986}{1000} = 98.6\%$

$recall(SPAM) = \frac{349}{363} = 96.1\%$ 

$precision(HAM) = \frac{986}{986+14} = 98.6\%$

$precision(SPAM) = \frac{349}{349+14} = 96.1\%$

The recall and precision for both class are pretty high which indicate that my classifier is powerful enough.

*<h3>Next Steps to be done</h3>*

* Try other smoothing methods
* Find a way to deal with the new words problem which I omitted this time
