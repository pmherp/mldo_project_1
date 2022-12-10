# Udacity Machine Learning DevOps Engineer - Project 1
## _Predict Customer Churn_

[Powered by: Udacity](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)

## Introduction
The Goal of this project is to create clean code that can be used in a production setting environment.
For this, Udacity provided the base code in form of a Jupyter Notebook, which was turned into a Python-Script that can be run with the command line.
This script is accompanied by a test script, which runs standard unit tests on the code and logging the results seperatly. 

## Installation

This project runs on python version 3.9 or higher. Make sure you have the right Python version installed on your local machine.

Install the dependencies and libraries with the requirements.txt provided in this repository:

```sh
pip install -r requirements.txt
```

This project uses the following libraries:
- astroid==2.12.13
- autopep8==2.0.0
- cloudpickle==2.2.0
- contourpy==1.0.6
- cycler==0.11.0
- dill==0.3.6
- fonttools==4.38.0
- isort==5.10.1
- joblib==1.2.0
- kiwisolver==1.4.4
- lazy-object-proxy==1.8.0
- llvmlite==0.39.1
- matplotlib==3.6.2
- mccabe==0.7.0
- numba==0.56.4
- numpy==1.23.5
- packaging==21.3
- pandas==1.5.2
- Pillow==9.3.0
- platformdirs==2.5.4
- pycodestyle==2.10.0
- pylint==2.15.8
- pyparsing==3.0.9
- python-dateutil==2.8.2
- pytz==2022.6
- scikit-learn==1.1.3
- scipy==1.9.3
- seaborn==0.12.1
- shap==0.41.0
- six==1.16.0
- slicer==0.0.7
- threadpoolctl==3.1.0
- tomli==2.0.1
- tomlkit==0.11.6
- tqdm==4.64.1
- typing_extensions==4.4.0
- wrapt==1.14.1

## How to run the Code
To run __churn_library.py__ that holds the entire Data Science Process Code, run the following in your terminal:
```sh
python3 churn_library.py
```
This will automatically run the entire code and log the results in a file named __test_results.log__ inside the folder __logs__.

To test the functions in __churn_library.py__, run the following in your terminal:
```sh
python3 churn_script_logging_and_tests.py
```
This will automatically test all functions in __churn_library.py__ with the provided unit tests and log the results in a file named __churn_library.log__ inside the folder __logs__.

## Results
The results of __churn_library.py__ are saved inside the folder __images__ which then contains png images for both the exploratoy data analysis as well as the model training results.

## License

MIT

I want to give credit for the data and basic code provide by Udacity's Machine Learning DevOps Nanodegree program.