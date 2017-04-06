import numpy as np
import pandas as pd
import re
import time
from string import punctuation

DEBUG = False


def print_time(t1, description):
    print(description)
    seconds = time.time() - t1
    print('{:.0f} hours  {:.0f} minutes'.format(seconds / 3600, (seconds / 60) % 60))
    print('--- {:08.6f} total seconds ---'.format(seconds), end='\n\n')


def clean_string(text):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", "", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"\bm\b", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r"\be g\b", " eg ", text)
    text = re.sub(r"\bb g\b", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"\b9 11\b", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r"\busa\b", " America ", text)
    text = re.sub(r"\bUSA\b", " America ", text)
    text = re.sub(r"\bu s\b", " America ", text)
    text = re.sub(r"\buk\b", " England ", text)
    text = re.sub(r"\bUK\b", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r"\bdms\b", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r"\bcs\b", " computer science ", text) 
    text = re.sub(r"\bupvotes\b", " up votes ", text)
    text = re.sub(r"\biPhone\b", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r"\bJ K\b", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    return text


# Execute as standalone script
if __name__ == '__main__':
    T1 = time.time()

    print_time(T1, 'Starting Script')

    # Load in the dataset
    train_df = pd.read_csv("../data/train.csv", low_memory=False)
    test_df = pd.read_csv("../data/test.csv", low_memory=False)

    # NOTE: Set to True if debugging
    if DEBUG:
        train_df = train_df.head(5)
        test_df = test_df.head(5)

    # Make sure that none of the questions are Null
    train_df['question1'].fillna('', inplace=True)
    train_df['question2'].fillna('', inplace=True)
    test_df['question1'].fillna('', inplace=True)
    test_df['question2'].fillna('', inplace=True)

    # Clean the question data
    print_time(T1, 'Cleaning training set, q1')
    train_df['question1'] = train_df['question1'].apply(lambda x: clean_string(x))
    print_time(T1, 'Cleaning training set, q2')
    train_df['question2'] = train_df['question2'].apply(lambda x: clean_string(x))
    print_time(T1, 'Cleaning test set, q1')
    test_df['question1'] = test_df['question1'].apply(lambda x: clean_string(x))
    print_time(T1, 'Cleaning test set, q2')
    test_df['question2'] = test_df['question2'].apply(lambda x: clean_string(x))

    print_time(T1, 'Writing output')

    train_df.to_csv('../cleaned_data/train_v0.csv', index=False)
    test_df.to_csv('../cleaned_data/test_v0.csv', index=False)

    print_time(T1, 'Done!')
