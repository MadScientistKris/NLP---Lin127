# Created by Yingxi Yu(913435445)

# Load president Washington's 1789 inaugural address
import nltk
import pandas as pd
from collections import Counter
from nltk.corpus import inaugural
Washington = inaugural.words('1789-Washington.txt')


# Count the total number of words, and report the number
number_of_words = len(Washington)
print(number_of_words)
# Here I took punctuations into count since punctuations are quite helpful
# to the analysis of speech text as you can see in question 5.

# Find the ten most frequent words and report them in a table with their counts
fdist = nltk.FreqDist(Washington)
ten_freq_table = pd.DataFrame(fdist.most_common(10))
ten_freq_table.columns = ['words', 'counts']
ten_freq_table

# Load president Obama's 1789 inaugural address
Obama = inaugural.words('2009-Obama.txt')
number_of_words2 = len(Obama)
print(number_of_words2)

fdist2 = nltk.FreqDist(Obama)
ten_freq_table2 = pd.DataFrame(fdist2.most_common(10))
ten_freq_table2.columns = ['words', 'counts']
print(ten_freq_table2)

# Do the same work without using NLTK
with open('1789-Washington.txt') as f:
    Washington2 = f.read()

Washington2_words = Washington2.split()
print(len(Washington2_words))
count = Counter(Washington2_words)
ten_freq_table3 = pd.DataFrame(count.most_common(10))
ten_freq_table3.columns = ['words', 'counts']
print(ten_freq_table3)
# We can see that The only difference between this outcome and the nltk outcome is the punctuations. 
# If we ignore the punctuations in the nltk outcome, the counts and most common words here are almost the same as in nltk