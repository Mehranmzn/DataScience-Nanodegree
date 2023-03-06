# Disaster Response Pipeline Project

## Table of Contents
 * [Project Motivation](#project-motivation)
 * [Installation](#Installation)
 * [File Descriptions](#file-descriptions)
 * [Components](#components)
 * [Instructions of How to Interact With Project](#instructions-of-how-to-interact-with-project)
 * [Licensing, Authors, Acknowledgements, etc.](#licensing-authors-acknowledgements-etc)
 
### Project Motivation
I utilized my data engineering expertise in this undertaking to scrutinize disaster data obtained from [Figure Eight](www.appen.com) and establish an API model that classifies disaster messages. To achieve this, I developed a machine learning pipeline capable of categorizing authentic messages transmitted during disaster occurrences. The aim of this project was to ensure that the messages get directed to the relevant disaster relief agency. Additionally, I created a web application that enables emergency workers to input new messages and obtain classification results across various categories. The web application also offers graphical representations of the data.

## Installation <a name="installation"></a>
- pandas
- re
- sys
- numpy
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3

### File Descriptions
app    

| - template    
| |- master.html # main page of web app    
| |- go.html # classification result page of web app    
|- run.py # Flask file that runs app    


data    

|- disaster_categories.csv # data to process    
|- disaster_messages.csv # data to process    
|- process_data.py # data cleaning pipeline    
|- InsertDatabaseName.db # database to save clean data to     


models   

|- train_classifier.py # machine learning pipeline     
|- classifier.pkl # saved model     


README.md    

### Components
There are three components completed for the project. 

#### 1. ETL Pipeline
A Python script, `process_data.py`, writes a data cleaning pipeline that:

 - Loads the messages.csv and categories.csv datasets in to the environement
 - Merges the two datasets
 - Cleans the data, Transforming
 - Stores it in a SQLite database
 
A jupyter notebook `ETL Pipeline Preparation` was used to do EDA (`process_data.py`). 
 
#### 2. ML Pipeline
A Python script, `train_classifier.py`, writes a machine learning pipeline:

 - Loads data from the SQLite database
 - Splits the dataset into training and test sets, tokenizing, remove NaNs
 - Builds a text processing and lammetizing for ML learning pipeline, 
 - Trains/Tunes the model using GridSearchCV by optimizing some of the parameters
 - Print scores/output of the model on the unseen test set
 - Exports the final model as a pickle file.
 
A jupyter notebook `ML Pipeline Preparation` was used to do EDA (`train_classifier.py`). 

#### 3. Flask Web App
Below are the outputs of the project's web application, which allows emergency workers to input new messages and receive classification results across multiple categories. The web app also presents data visualizations:

![app2](https://github.com/Mehranmzn/DataScience-Nanodegree/blob/master/notebooks/Project%202/app/98724735-159df880-238c-11eb-8338-bc4b4e0b1c39.JPG)





![app2](https://github.com/Mehranmzn/DataScience-Nanodegree/blob/master/notebooks/Project%202/app/98724932-5bf35780-238c-11eb-8a93-ebb09ab2d510.JPG)


### Instructions of How to Interact With Project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements, etc.
Thanks to Udacity for starter code for the web app. 
