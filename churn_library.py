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


from sklearn.metrics import classification_report
import sklearn.metrics as metrics

import logging

logging.basicConfig(
        filename='./logs/churn_library.log',
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
    #read in csv file from given path and store in pandas dataframe
    df = pd.read_csv(pth)

    return df

def generate_churn_column(df):
    '''
    generates new column "churn" to predict on.

    input:
        df: (pd.DataFrame)
    output:
        df: (pd.DataFrame)
    '''
    #generate output column "churn"
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    return df
    

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
    #seperate categorical and object columns and put into list
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    #seperate numeric columns and put into list
    quant_columns = df.select_dtypes(include=['number']).columns.tolist()
    #drop unneeded columns from quant columns list
    quant_columns = quant_columns[2:]

    return cat_columns, quant_columns


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
    plt.figure('Figure 4', figsize=(20,10)) 
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
    #get shape of df
    print(f'The dataframe has {df.shape[0]} rows and {df.shape[1]} columns')

    #get sum of empty rows for each column
    print(f'Sum of empty rows for each column in df:\n{df.isnull().sum()}')

    #get description for df
    print(f'Basic description for each column in df:\n{df.describe()}')

    #generate hist plot for churn column in df
    generate_hist_plot(df['Churn'], './images/eda/churn_histplot.png')
        
    #generate hist plot for customer_age in df
    generate_hist_plot(df['Customer_Age'], './images/eda/customer_age_histplot.png')

    #generate bar chart for marital status
    plt.figure('Figure 5', figsize=(10, 8)) 
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_barchart.png')

    #Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
    plt.figure('Figure 6', figsize=(10, 8)) 
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/total_trans_ct_histplot.png')

    #generate heatmap correlation plot
    plt.figure('Figure 7', figsize=(10, 8)) 
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('./images/eda/correlation_plot.png')


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
    for item in category_lst:
        lst = []
        group = df.groupby(item).mean(numeric_only=True)[response]

        for val in df[item]:
            lst.append(group.loc[val])

        name = item + '_' + response
        df[name] = lst

    return df

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
    plt.rc('figure', figsize=(10, 8))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    #save plot
    plt.savefig(f'{output_pth}random_forrest_classification_report.png')

    plt.rc('figure', figsize=(10, 8))
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
    #create figure 1
    plt.figure('Figure 1', figsize=(10, 8))
    #generate shap summary_plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    #save plot
    plt.savefig(f'{output_pth}shap_summary.png')

    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    #create figure 2
    plt.figure('Figure 2', figsize=(10, 8))

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
    #create figure 2
    plt.figure('Figure 3', figsize=(10, 8))
    fpr_rf, tpr_rf, threshold = metrics.roc_curve(y_test, y_test_preds_rf)
    fpr_lr, tpr_lr, threshold = metrics.roc_curve(y_test, y_test_preds_lr)
    roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)
    roc_auc_lr = metrics.auc(fpr_lr, tpr_lr)
    ax = plt.axes()
    ax.set_facecolor('silver')
    plt.plot(fpr_rf, tpr_rf, label = 'AUC = %0.2f' % roc_auc_rf, color='blue')
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