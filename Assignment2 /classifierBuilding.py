import nltk
import pickle


def get_freq_df(words):
    '''Build a pandas data frame of the frequency table
    Args:
        words(nltk object): a list of tokens
    Returns:
        dist_words(nltk FreqDist): frequency table
    '''
    dist_words = nltk.FreqDist(words)
    return dist_words



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


if __name__ == '__main__':
  # path = '/Users/Aslan/winter 2017/Lin127/hw2/SPAM_training'
    path = './SPAM_training' 
    ham = nltk.corpus.PlaintextCorpusReader(path, 'HAM.*.txt')
    spam = nltk.corpus.PlaintextCorpusReader(path, 'SPAM.*.txt')
    
    ham_count = len(ham.fileids())
    spam_count = len(spam.fileids())
    
    ham_tokens = ham.words()
    spam_tokens = spam.words()

    solver_NB = solver(spam_tokens, ham_tokens, spam_count, ham_count)
    with open('spam.nb', 'wb') as f:
        pickle.dump(solver_NB, f, protocol=pickle.HIGHEST_PROTOCOL)
