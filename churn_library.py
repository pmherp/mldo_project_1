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

from scikitplot.metrics import plot_roc
from sklearn.metrics import classification_report
import sklearn.metrics as metrics

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

def generate_churn_column(df):
    '''
    generates new column "churn" to predict on.

    input:
        df: (pd.DataFrame)
    output:
        df: (pd.DataFrame)
    '''
    try:
        #test if "Attrition_Flag" column exists in df
        assert df.columns.isin(['Attrition_Flag']).any()

        #logging for success
        logging.info('SUCCESS: Attrition_Flag column exists. Able to generate churn column')

        #generate output column "churn"
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

        return df
    except AssertionError:
        logging.error('ERROR: Attrition_Flag column does not exist. Unable to generate churn column.')

        return None
    

def split_cat_quant_cols(df):
    '''
    splits columns into categorical and quantitive.
    stores columns in seperate lists.

    input:
        df: (pd.DataFrame)
    output:
        cat_columns: (list)
        quant_columns: (list)
    '''
    try:
        #test if columns exist in input df
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

        #loging message for success
        logging.info('SUCCESS: all columns are in DataFrame. Split into categorical and quantitive is possible.')

        #seperate categorical and object columns and put into list
        cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        #seperate numeric columns and put into list
        quant_columns = df.select_dtypes(include=['number']).columns.tolist()
        #drop unneeded columns from quant columns list
        quant_columns = quant_columns[2:]

        return cat_columns, quant_columns
    except AssertionError:
        logging.error('ERROR: At least one column is missing from DataFrame. Split into categorical and quantitive is not possible.')

        return None

    

def generate_hist_plot(column, pth):
    '''
    creates and saves histplot.

    input:
        column: (pd.DataFrame.column)
        pth: (str) path to save plot image as png
    output:
        None
    '''
    #generate hist plot for churn column in df
    plt.figure(figsize=(20,10)) 
    column.hist()
    #save churn hist plot in ./images/eda
    plt.savefig(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder.

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
        logging.error('ERROR: wrong path to save plots into. Columns "Churn", "Customer_Age", "Marital_Status" or "Total_Trans_Ct" not in DataFrame.')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook.

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used for naming variables or index y column]
    output:
        df: pandas dataframe with new columns for
    '''
    try:
        for item in category_lst:
            lst = []
            group = df.groupby(item).mean()[response]

            for val in df[item]:
                lst.append(group.loc[val])

            name = item + '_' + response
            df[name] = lst

        return df
    except:
        return None

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
    #filter only category columns and put into list
    category_lst, quant_lst = split_cat_quant_cols(df)

    #final features list
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    
    #perform encoding for category columns
    df = encoder_helper(df, category_lst, response)

    #define output variable
    y = df[response]

    #define input variables
    X = pd.DataFrame()
    
    X[keep_cols] = df[keep_cols]

    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_pth):
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
    plt.rc('figure', figsize=(5, 5))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    #save plot
    plt.savefig(f'{output_pth}random_forrest_classification_report.png')

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    #save plot
    plt.savefig(f'{output_pth}logistic_regression_classification_report.png')



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
    #explainer = shap.TreeExplainer(model.best_estimator_)
    #shap_values = explainer.shap_values(X_data)
    #shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    #save plot
    #plt.safefig(f'{output_pth}shap_values.png')

    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    #save plot
    plt.savefig(f'{output_pth}feature_importance.png')
    

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
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(max_iter=3000)

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    #plot for AUC
    #plot_roc_curve is depricated since scikit version 0.23
    #alternative
    fpr_rf, tpr_rf, threshold = metrics.roc_curve(y_test, y_test_preds_rf)
    fpr_lr, tpr_lr, threshold = metrics.roc_curve(y_test, y_test_preds_lr)
    roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)
    roc_auc_lr = metrics.auc(fpr_lr, tpr_lr)
    plt.plot(fpr_rf, tpr_rf, 'b', label = 'AUC = %0.2f' % roc_auc_rf)
    plt.plot(fpr_lr, tpr_lr, label = 'AUC = %0.2f' % roc_auc_lr, color='orange')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./images/results/roc_curve_plot.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    #create feature importance plot and save it
    feature_importance_plot(cv_rfc.best_estimator_, X_train, './images/results/')

    #create classification report for each model
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                './images/results/')

if __name__ == '__main__':
    #load data
    logging.info('INFO: Loading data.')
    df = import_data()

    #generate output column "Churn"
    logging.info('INFO: Generate Churn column as output.')
    df = generate_churn_column(df)

    #perform exploratory data analysis
    logging.info('INFO: Do Exploratory Data Anaylsis')
    perform_eda(df)

    #perform feature engineering
    logging.info('INFO: Performing feature engineering.')
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

    #train models
    #generate feature importance plot
    #generate classification report
    logging.info('INFO: Training models. Generating Feature Importance plot. Generating Classification Report.')
    train_models(X_train, X_test, y_train, y_test)