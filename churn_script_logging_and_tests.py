"""
Python Script including tests for the Data Science Process defined in churn_libary.py for:
- Importing
- Exploratory Data Analysis
- Feature Engineering
- Model Training
- Model Evaluation

Author: pmherp
Date: 10.12.2022
"""

import os
import logging
import pandas as pd
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions

    inputs:
            - import_data: (function) imports data from path
    returns:
            - df: (pd.DataFrame)
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("SUCCESS: Testing import_data function.")
    except FileNotFoundError as err:
        logging.error("ERROR testing import_data: The file wasn't found")
        raise err

    try:
        # check if input df is empty
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        # check if given path is of dtype str
        assert isinstance("./data/bank_data.csv", str)
        # check if given path exists and has input file in it
        assert os.path.exists("./data/bank_data.csv")
    except AssertionError as err:
        logging.error(
            "ERROR testing import_data: The file doesn't appear to have rows and columns. Input type not string. Filepath not found.")
        raise err

    return df


def test_generate_churn_column(generate_churn_column, df):
    '''
    test perform eda function

    inputs:
            - generate_churn_column: (function) generating output column for ML
            - df: (pd.DataFrame)
    returns:
            - df: (pd.DataFrame) with output column
    '''
    df = generate_churn_column(df)

    try:
        # check if column Attrition_Flag exists in input data
        assert set(['Attrition_Flag']).issubset(df.columns)
        logging.info('SUCCESS: testing generate_churn_column.')
    except AssertionError as err:
        logging.error(
            'ERROR testing generate_churn_column: Attrition_Flag column not in input data.')
        raise err

    return df


def test_generate_hist_plot(generate_hist_plot, df):
    '''
    Testing function to creat histogram plot

    inputs:
            - generate_hist_plot: (function) that plots histogras
            - df: (pd.DataFrame)
    returns:
            - None
    '''
    generate_hist_plot(df['Churn'], './images/eda/churn_histplot.png')
    generate_hist_plot(
        df['Customer_Age'],
        './images/eda/customer_age_histplot.png')

    try:
        # check if columns Churn and Customer_Age exist in input data
        assert set(['Churn', 'Customer_Age']).issubset(df.columns)
        # check if path exists
        assert os.path.exists('./images/eda/')
        logging.info('SUCCESS: testing generate_hist_plot.')
    except AssertionError as err:
        logging.error(
            'ERROR testing generate_hist_plot: Columns "Churn" or "Customer_Age" not in input data. Path to save plots does not exist.')
        raise err


def test_eda(perform_eda, df):
    '''
    test perform eda function

    inputs:
            - perform_eda: (function) all eda operations in one function plotting the results
            - df: (pd.DataFrame)
    returns:
            - None
    '''
    perform_eda(df)

    try:
        # test for df dtype DataFrame
        assert isinstance(df, pd.DataFrame)
        # test for whether path exists
        assert os.path.exists('./images/eda')
        # test if specific columns are in input data
        assert df.columns.isin(
            ['Churn', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct']).any()
        logging.info('SUCCESS: testing perform_eda.')
    except AssertionError as err:
        logging.error('ERROR testing perform_eda: input data not of type pd.DataFrame. Image path not found. Columns "Churn", "Customer_Age", "Marital_Status" or "Total_Trans_Ct" not in input data.')
        raise err


def test_split_cat_quant_cols(split_cat_quant_cols, df):
    '''
    Test split cat_quant_cols function for splitting categories and number columns

    inputs:
            - split_cat_quant_cols: (function) seperating numerical and categorical columns
            - df: (pd.DataFrame)
    returns:
            - category_lst: (list) with categorical columns
            - quant_lst: (list) with numerical columns
    '''
    category_lst, quant_lst = split_cat_quant_cols(df)

    try:
        # test if needed columns are in input data
        assert set(['Gender',
                    'Education_Level',
                    'Marital_Status',
                    'Income_Category',
                    'Card_Category',
                    'Customer_Age',
                    'Dependent_count',
                    'Months_on_book',
                    'Total_Relationship_Count',
                    'Months_Inactive_12_mon',
                    'Contacts_Count_12_mon',
                    'Credit_Limit',
                    'Total_Revolving_Bal',
                    'Avg_Open_To_Buy',
                    'Total_Amt_Chng_Q4_Q1',
                    'Total_Trans_Amt',
                    'Total_Trans_Ct',
                    'Total_Ct_Chng_Q4_Q1',
                    'Avg_Utilization_Ratio']).issubset(df.columns)

        logging.info('SUCCESS: testing split_cat_quant_cols function.')
    except AssertionError as err:
        logging.info(
            'ERROR testing split_cat_quant_cols function: columns not in input data.')
        raise err

    return category_lst, quant_lst


def test_encoder_helper(encoder_helper, category_lst, df):
    '''
    test encoder helper

    inputs:
            - encoder_helper: (function) encoding all categorical columns
            - category_lst: (list) with categorical columns
            - df: (pd.DataFrame)
    returns:
            - df: (pd.DataFrame)
    '''
    df = encoder_helper(df, category_lst, 'Churn')

    try:
        # check if category list is empty
        assert len(category_lst) != 0
        logging.info('SUCCESS: testing encoder_helper function')
    except AssertionError as err:
        logging.error(
            'ERROR testing encoder_helper function: category list is empty.')
        raise err

    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering

    inputs:
            - perform_feature_engineering: (function) split wanted input/output data for training
            - df: (pd.DataFrame)
    returns:
            - X_train: (pd.Series) train input values
            - X_test: (pd.Series) test input values
            - y_train: (np.array) train output values
            - y_test: (np.array) test output values
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

    try:
        # check input data (splits) has data
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info('SUCCESS: testing perform_feature_engineering function.')
    except AssertionError as err:
        logging.error(
            'ERROR testing perform_feature_engineering function: input of train, test splits empty.')
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models

    inputs:
            - train_models: (function) that trains specified models
            - X_train: (pd.Series) train input values
            - X_test: (pd.Series) test input values
            - y_train: (np.array) train output values
            - y_test: (np.array) test output values
    returns:
            - None
    '''
    train_models(X_train, X_test, y_train, y_test)

    try:
        # check if path to save models exists
        assert os.path.exists('./models/rfc_model.pkl')
        logging.info(
            'SUCCESS: testing train_models function for dump rfc_model.')
    except AssertionError as err:
        logging.error(
            'ERROR testing train_model function: path to store rfc_model does not exist.')
        raise err

    try:
        # check if path to save models exists
        assert os.path.exists('./models/logistic_model.pkl')
        logging.info(
            'SUCCESS: testing train_models function for dump logistic_model.')
    except AssertionError as err:
        logging.error(
            'ERROR testing train_model function: path to store logistic_model does not exist.')
        raise err


if __name__ == "__main__":
    # test for data import function
    DF = test_import(cls.import_data)

    # test generating output column function
    DF = test_generate_churn_column(cls.generate_churn_column, DF)

    # test perform exploratory data analysis function
    test_eda(cls.perform_eda, DF)

    # test for feature engineering function
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, DF)

    # test train models function
    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
