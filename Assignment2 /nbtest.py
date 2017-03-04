import nltk
import pickle
import math


def spam_detection(file, ham_log_likelihood, spam_log_likelihood,
                   log_prior_spam, log_prior_ham, vocabulary, test):

    token_ls = test.words(fileids=file)
    log_posterior_spam = log_prior_spam
    log_posterior_ham = log_prior_ham

    for token in token_ls:
        if token not in vocabulary:
            continue
        log_posterior_spam += spam_log_likelihood[token]
        log_posterior_ham += ham_log_likelihood[token]

    if log_posterior_spam > log_posterior_ham:
        print(file, 'SPAM')
    else:
        print(file, 'HAM')


if __name__ == '__main__':
    with open('spam.nb', 'rb') as f:
        model = pickle.load(f)
    log_prior_spam = math.log(model['spam_count'] / (model['spam_count'] +
                              model['ham_count']))
    log_prior_ham = math.log(model['ham_count'] / (
                               model['spam_count'] + model['ham_count']))
    spam_n = model['spam_fd'].N()
    ham_n = model['ham_fd'].N()

    vocabulary = set(list(model['ham_fd'].keys()) + list(model['spam_fd'].keys()))
    vocal_size = len(vocabulary)

    spam_log_likelihood = {}
    ham_log_likelihood = {}

    for token in vocabulary:
        spam_log_likelihood[token] = math.log((model['spam_fd'][token] + 1) / (spam_n + vocal_size))
        ham_log_likelihood[token] = math.log((model['ham_fd'][token] + 1) / (ham_n + vocal_size))

    test = nltk.corpus.PlaintextCorpusReader('/Users/Aslan/winter 2017/Lin127/hw2,3/SPAM_dev', r'.*txt')

    for email in test.fileids():
        spam_detection(email, ham_log_likelihood, spam_log_likelihood,
                       log_prior_spam, log_prior_ham, vocabulary, test)
