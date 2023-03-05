# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import pickle




nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    Loads data from SQLite database.
    
    Parameters:
    database_filepath: Filepath to the database
    
    Returns:
    X: Features
    Y: Target
    """
    # load data from database 
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_response", con=engine)
    df = df[df['related'] != 2]
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y


def tokenize(text):
    """
    Tokenizes and lemmatizes text.
    
    Parameters:
    text: Text to be tokenized
    
    Returns:
    clean_tokens: Returns cleaned tokens 
    """
    tokens = word_tokenize(text.lower()) # Convert to lowercase and tokenize

    
    lemmatizer = WordNetLemmatizer()

    
    clean_tokens = [lemmatizer.lemmatize(tok) for tok in tokens if re.match(r'\w+', tok)] # Lemmatize each word and remove non-alphanumeric characters
    return clean_tokens


def build_model():
    """
    Builds classifier and tunes model using GridSearchCV.
    
    Returns:
    cv: Classifier 
    """    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize, lowercase=True)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42), n_jobs=-1))
    ])
        
    parameters = {
        'clf__estimator__max_depth': [5, 20]
     
     }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of model and returns classification report. 
    
    Parameters:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test
    
    Returns:
    Classification report for each column
    
    """
    y_pred = model.predict(X_test)
    
    for col in range(y_test.shape[1]):
        print("=======================",col,"======================")

        print(classification_report(y_pred[col], y_test[col]))

    


    
def save_model(model, model_filepath):
    """ Exports the final model as a pickle file."""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Builds the model, trains the model, evaluates the model, saves the model."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        # create a SimpleImputer object
        imputer = SimpleImputer(strategy='mean')

        # fit the imputer on X_train
        imputer.fit(y_train)

        # transform 
        y_train = imputer.transform(y_train).round(0)
        y_test = imputer.transform(y_test).round(0)


        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
