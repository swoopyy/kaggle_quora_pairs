import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
import time


# Define some basic helper functions
def extract_basic_features(df):
    feature_df = pd.DataFrame()
    feature_df['q1_cLen'] = df['question1'].apply(len)
    feature_df['q2_cLen'] = df['question2'].apply(len)
    feature_df['q1_wLen'] = df['question1'].apply(lambda x: len(x.split()))
    feature_df['q2_wLen'] = df['question2'].apply(lambda x: len(x.split()))
    return feature_df


def print_time(t1, description):
    print(description)
    seconds = time.time() - t1
    print('{:.0f} hours  {:.0f} minutes'.format(seconds / 3600, (seconds / 60) % 60))
    print('--- {:08.6f} total seconds ---'.format(seconds), end='\n\n')


# Execute as standalone script
if __name__ == '__main__':
    T1 = time.time()

    print_time(T1, 'Starting Script')

    # Load in the dataset
    train_df = pd.read_csv("../cleaned_data/train_v0.csv", low_memory=False)
    test_df = pd.read_csv("../cleaned_data/test_v0.csv", low_memory=False)

    # NOTE: Set to True if debugging
    if False:
        train_df = train_df.head(100)
        test_df = test_df.head(100)

    # Make sure that none of the questions are Null
    train_df['question1'].fillna('', inplace=True)
    train_df['question2'].fillna('', inplace=True)
    test_df['question1'].fillna('', inplace=True)
    test_df['question2'].fillna('', inplace=True)

    # Add all the questions together for LSA
    all_questions = [
        x for x in
        list(train_df['question1']) +
        list(train_df['question2']) +
        list(test_df['question1']) +
        list(test_df['question2'])
    ]

    print_time(T1, 'Data Loaded')

    # Build the tf-idf transformer from the list of all questions
    tfidf = TfidfVectorizer(stop_words='english', max_features=20000)
    tfidf_transformer = tfidf.fit(all_questions)

    print_time(T1, 'tf-idf transformer constructed')

    # Find the transformation from tf-idf space using SVD to reduced space of
    # only 100 dimensions, aka latent semantic analysis
    svd = TruncatedSVD(n_components=100, algorithm='arpack')
    svd_transformer = svd.fit(tfidf_transformer.transform(all_questions))

    print_time(T1, 'svd transformer constructed')

    # Transform the individual questions into our reduced space via LSA
    train_lsa_q1 = svd_transformer.transform(
        tfidf_transformer.transform(train_df['question1'])
    )
    train_lsa_q2 = svd_transformer.transform(
        tfidf_transformer.transform(train_df['question2'])
    )
    test_lsa_q1 = svd_transformer.transform(
        tfidf_transformer.transform(test_df['question1'])
    )
    test_lsa_q2 = svd_transformer.transform(
        tfidf_transformer.transform(test_df['question2'])
    )

    print_time(T1, 'question transformation complete')

    # Calculate the dot products of the lsa transformed questions in the training set
    train_dp = [
        train_lsa_q1[i].dot(train_lsa_q2[i].T)
        for i in range(len(train_df))
    ]

    print_time(T1, 'training dot products complete')

    # Calculate the dot products on the lsa transformed questions in the test
    test_dp = [
        test_lsa_q1[i].dot(test_lsa_q2[i].T)
        for i in range(len(test_df))
    ]

    print_time(T1, 'testing dot products complete')

    # Set up feature set for training and test sets
    X = extract_basic_features(train_df)
    X['dot_products'] = train_dp
    test_X = extract_basic_features(test_df)
    test_X['dot_products'] = test_dp

    print_time(T1, 'feature set extraction complete')

    # Extract targets from training set
    Y = train_df['is_duplicate'].values

    # Train the classifiers
    lr_clf = LogisticRegression()
    rf_clf = RandomForestClassifier(n_estimators=25)
    et_clf = ExtraTreesClassifier(n_estimators=25)

    vclf = VotingClassifier(
        [('lr', lr_clf), ('rf', rf_clf), ('et', et_clf)],
        voting='soft'
    ).fit(X, Y)

    print_time(T1, 'classifier training complete')

    # Form predictions for test data and return the probability that it belongs
    # to each class, as specified by the kaggle guidelines for this contest
    predictions = vclf.predict_proba(test_X)

    # Create new dataframe to store results and export to csv
    submission = pd.DataFrame()
    submission['test_id'] = test_df['test_id']
    # predictions[:,0] is the probability that they're not duplicates
    # predictions[:,1] is the probability that they are duplicates,
    # which is what we want
    submission['is_duplicate'] = [x[1] for x in predictions]

    print_time(T1, 'Writing output')

    submission.to_csv('../submissions/submission_3.csv', index=False)

    print_time(T1, 'Done!')
