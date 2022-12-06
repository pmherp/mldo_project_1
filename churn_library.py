# library doc string


# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

import logging

logging.basicConfig(
        filename='./logs/test_results.log',
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s'
)

def import_data(pth='./data/bank_data.csv'):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: (str) a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        #test for filepath dtype string
        assert isinstance(pth, str)

        #test if file exists in path
        assert os.path.exists(pth)

        #log success messages
        logging.info('SUCCESS: fileinput is of type string and file exists at given path')

        #read in csv file from given path and store in pandas dataframe
        df = pd.read_csv(pth)

        return df
    except AssertionError:
        logging.error('ERROR: input is not of dtype string or file does not exist')

        return None

def generate_hist_plot(column, pth):
    '''
    creates and saves histplot
    input:
        column: (pd.DataFrame.column)
        pth: (str) path to save plot image as png
    '''
    #generate hist plot for churn column in df
    plt.figure(figsize=(20,10)) 
    column.hist()
    #save churn hist plot in ./images/eda
    plt.savefig(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    try:
        #test for df dtype DataFrame
        assert isinstance(df, pd.DataFrame)

        #logging success message
        logging.info('SUCCESS: df is of type pd.DataFrame')

        #get shape of df
        print(f'The dataframe has {df.shape[0]} rows and {df.shape[1]} columns')

        #get sum of empty rows for each column
        print(f'Sum of empty rows for each column in df:\n{df.isnull().sum()}')

        #get description for df
        print(f'Basic description for each column in df:\n{df.describe()}')
    except AssertionError:
        logging.error('ERROR: getting shape of dataframe failed. Wrong dtype.')

    try:
        #test if "Attrition_Flag" column exists in df
        assert df.columns.isin(['Attrition_Flag']).any()

        #logging for success
        logging.info('SUCCESS: Attrition_Flag column exists. Able to generate churn column')

        #generate output column "churn"
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    except AssertionError:
        logging.error('ERROR: Attrition_Flag column does not exist. Unable to generate churn column.')
    
    try:
        #test for whether path exists
        assert os.path.exists('./images/eda')

        #test for column needed to plot churn
        assert df.columns.isin(['Churn', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct']).any()

        #success message if path exists
        logging.info('SUCCESS: path to store image in exists and column needed to plot too')

        #generate hist plot for churn column in df
        generate_hist_plot(df['Churn'], './images/eda/churn_histplot.png')
        
        #generate hist plot for customer_age in df
        generate_hist_plot(df['Customer_Age'], './images/eda/customer_age_histplot.png')

        #generate bar chart for marital status
        plt.figure(figsize=(20,10)) 
        df['Marital_Status'].value_counts('normalize').plot(kind='bar')
        plt.savefig('./images/eda/marital_status_barchart.png')

        #Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
        plt.figure(figsize=(20,10)) 
        sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
        plt.savefig('./images/eda/total_trans_ct_histplot.png')

        #generate heatmap correlation plot
        plt.figure(figsize=(20,10)) 
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        plt.savefig('./images/eda/correlation_plot.png')
    except AssertionError:
        logging.error('ERROR: wrong path to save plot into. Columns "Churn", "Customer_Age", "Marital_Status", "Total_Trans_Ct" not in DataFrame.')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass