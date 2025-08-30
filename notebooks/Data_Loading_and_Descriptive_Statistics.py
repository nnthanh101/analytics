#!/usr/bin/env python
# coding: utf-8

# <header style="padding:1px;background:#f9f9f9;border-top:3px solid #00b2b1"><img id="Teradata-logo" src="https://www.teradata.com/Teradata/Images/Rebrand/Teradata_logo-two_color.png" alt="Teradata" width="220" align="right" />
# 
# <b style='font-size:28px;font-family:Arial;color:#E37C4D'>🎓 Predicting Air Particulate Matter at Scale ⛅️</b><br>
# <b style='font-size:28px;font-family:Arial;color:#E37C4D'>🛠️ [Reusable] Data Loading & Descriptive Statistics</b>
# </header>

# <div style="padding:0px; 
#             color:white;
#             margin:0px;
#             font-size:200%;
#             text-align:left;
#             display:fill;
#             border-radius:0px;
#             border-width: 5px;
#             border-style: solid;
#             border-color: green;
#             background-color:green;
#             overflow:hidden;
#             font-weight:600">🦅 Overview</div>
# 
# <div class="alert alert-block alert-success">
# 
# <p>🎓 This notebook loads air quality data and meteorological data from the [Auckland Council Environmental Data Portal](https://environmentauckland.org.nz/Data/Dashboard/183). Afterwards, Descriptive Statistics describes the characteristics of a data set. It should be ready for reuse in the next steps in CRISP-DM for Data Science</p>
#     
# </div>

# * **Data:**
# 
#   * [x] Six key air pollutants from WHO’s standard, including PM10 and PM2.5 (particulate matter with diameters of 10 and 2.5 microns), ozone (O3), nitrogen dioxide (NO2), sulphur dioxide (SO2), and carbon monoxide (CO).
#   * [x] Meteorological/Weather factors: wind speed, wind direction, air temperature, and relative humidity are crucial to determining the spread and behaviour of PM2.5/PM10 pollutants.
# 
# * **Workflow Steps:**
# 
#   1. Import the required libraries, including [teradataml](https://pypi.org/project/teradataml).
#   2. [Taradata] Connect to a Vantage system.
#   3. Data Loading from *.csv
#   4. [Teradata] Data Loading in Teradata Vantage
#   5. Descriptive Statistics

# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
#     
# # 🎯 Libraries and Reusable Functions 

# <div class="alert alert-block alert-info">
# 🎓 In the section, we import the required libraries and set environment variables and environment paths (if required).
# </div>

# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Import the required libraries

# In[1]:


# %%capture
# !pip install -r requirements.txt


# In[2]:


## Data Manipulation and Computation
import pandas as pd              ## Data processing, file I/O
import numpy as np               ## Linear algebra
from scipy import stats
from scipy.stats import skew, kurtosis, boxcox
from scipy.stats.mstats import winsorize, pearsonr

## Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot

## Web Application Framework (for Dash)
import dash
# import dash_core_components as dcc
# import dash_html_components as html
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import os, time, math, logging
# import textwrap
from dotenv import load_dotenv

## Time Series Analysis
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from pmdarima import preprocessing
# from pmdarima.arima import auto_arima

## Machine Learning - Model Building and Evaluation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
# from xgboost import XGBRegressor

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.models import Sequential
# from keras.optimizers import Adam
# # from tensorflow.keras.layers import Dense, LSTM
# # from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# # from tensorflow.keras.callbacks import EarlyStopping

import warnings ## Supress warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")
# warnings.simplefilter(action='ignore', category=FutureWarning)
# display.max_rows=5


# In[3]:


import getpass
from collections import OrderedDict

from teradataml import db_list_tables
import teradataml as tdml
from teradataml import read_csv, FillNa, valib
from teradataml import db_drop_table, fastload, OneHotEncodingFit, OneHotEncodingTransform, ScaleFit, ScaleTransform
from teradataml.context.context import *
from teradataml.dataframe.copy_to import copy_to_sql
from teradataml.dataframe.dataframe import DataFrame as TeradataDataFrame
from teradataml.options.configure import configure
from teradatasqlalchemy import (INTEGER, FLOAT, VARCHAR)

import json
from sqlalchemy import func


# In[44]:


# import os, logging
# from dotenv import load_dotenv
from prettytable import PrettyTable

## Configure logging to handle lower levels; setting level to DEBUG will show all subsequent levels.
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_environment_settings_as_table():
    """
    Prints the values of the specified environment variables in a table format.

    :param env_vars: A dictionary of environment variable names and their descriptions.
    """
    ## Dictionary mapping environment variable names to their descriptions
    environment_variables = {
        'IS_LOADING_FROM_FILES': 'True if loading data from *.csv/xls files; False if using imported data in Teradata Vantage',
        'IS_TERADATA_VANTAGE': 'Using scable Teradata Vantage vs. local machine (Laptop)',
        'IS_DATA_IN_TERADATA_VANTAGE': 'Using TeradataDataFrame in scalable Vantage vs. PandasDataFrame with local *.csv/xls files',
        'SCHEMA_NAME': '[Teradata Vantage] Schema Name',
        'TABLE_NAME': '[Teradata Vantage] Table Name',
        'IS_JUPYTERLAB': 'Running in JupyterLab vs Python Dash/Vizro Dashboard',
        # 'IS_DASHBOARD': 'Using a stand-alone Dashboard UI/UX (not JupyterLab)',
        'IS_TEST_DEV': 'Is Test/Dev mode is active or not (in Production)',
        'DATA_PATH':'*.csv/xls Data PATH',
        'USE_DATA_PREFIX': "Prefix to use for data files: 'raw' | 'cleaned'",
    }

    table = PrettyTable()  ## Create a PrettyTable object with specified column names
    table.field_names = ["Variable", "Description", "Value"]
    table.align = "l"      ## Align the text to the left

    ## Loop through each environment variable and its value in the environment_variables | env_vars dictionary
    for var, description in environment_variables.items():
        value = os.getenv(var, 'Not set or not found')
        ## Get the description from the descriptions dictionary; default to 'No description provided' if not found
        description = environment_variables.get(var, 'No description provided')
        ## Add a row to the table with the variable name, its description, and its value
        table.add_row([var, description, value])

    print(table)


def load_environment_settings(env_path_default='cleaned_vantage.env', is_print_table=True):
    """
    Loads environment variables from a .env file and sets up global variables for the application.
    
    Parameters:
    - env_path_default (str): Default path to the .env file if ENV_PATH is not set in the environment.
    
    Returns:
    - A dictionary containing the loaded and computed environment settings.
    - This function sets global variables that can be used throughout the application.
    """
    ## Load .env file specified by the ENV_PATH or use the default path
    dotenv_path = os.getenv('ENV_PATH', env_path_default)
    load_dotenv(dotenv_path=dotenv_path)
    print(f'🎓 The specified .env file: {dotenv_path}')

    ## Set global variables based on the environment variables 
    global IS_JUPYTERLAB, IS_LOADING_FROM_FILES, IS_TERADATA_VANTAGE, IS_DATA_IN_TERADATA_VANTAGE, IS_TEST_DEV, SCHEMA_NAME, TABLE_NAME, USE_DATA_PREFIX, DATA_PATH

    ## Access environment variables as if they came from environment settings in env_path_default='cleaned_vantage.env'
    IS_JUPYTERLAB               = os.getenv('IS_JUPYTERLAB') == 'True'
    IS_LOADING_FROM_FILES       = os.getenv('IS_LOADING_FROM_FILES') == 'True'
    IS_TERADATA_VANTAGE         = os.getenv('IS_TERADATA_VANTAGE') == 'True'
    IS_DATA_IN_TERADATA_VANTAGE = os.getenv('IS_DATA_IN_TERADATA_VANTAGE') == 'True'
    IS_TEST_DEV                 = os.getenv('IS_TEST_DEV') == 'True'
    USE_DATA_PREFIX             = os.getenv('USE_DATA_PREFIX')
    SCHEMA_NAME                 = os.getenv('SCHEMA_NAME')
    TABLE_NAME                  = os.getenv('TABLE_NAME')
    DATA_PATH                   = f'data/{USE_DATA_PREFIX}'   ## 'raw' | 'cleaned'
    # IS_DASHBOARD                = os.getenv('IS_DASHBOARD') == 'True'

    ## Call the function to print the environment variables
    if is_print_table:
        print_environment_settings_as_table()

    config_env_settings = {
        'IS_JUPYTERLAB':               IS_JUPYTERLAB,
        'IS_LOADING_FROM_FILES':       IS_LOADING_FROM_FILES,
        'IS_TERADATA_VANTAGE':         IS_TERADATA_VANTAGE,
        'IS_DATA_IN_TERADATA_VANTAGE': IS_DATA_IN_TERADATA_VANTAGE,
        'IS_TEST_DEV':                 IS_TEST_DEV,
        'USE_DATA_PREFIX':             USE_DATA_PREFIX,     ## Default prefix
        'SCHEMA_NAME':                 SCHEMA_NAME,
        'TABLE_NAME':                  TABLE_NAME,
        'DATA_PATH':                   DATA_PATH,           ## Construct the data path
    }
    return config_env_settings


# In[47]:


## Python usage
# if __name__ == "__main__":
#     load_environment_settings(env_path_default='cleaned_vantage.env')

# ## .env --> Setting the environment variable for output format programmatically
env_settings = load_environment_settings(env_path_default='cleaned_dashboard.env')


# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Connect to Vantage
# 
# You will be prompted to provide the `password`. Enter your password, press the Enter key, then use down arrow to go to next cell.

# In[6]:


# if IS_TERADATA_VANTAGE:
#     get_ipython().run_line_magic('run', '-i ../startup.ipynb')
#     eng = create_context(host = 'host.docker.internal', username='demo_user', password = password)
#     logging.info(eng)


# In[43]:


# if IS_TERADATA_VANTAGE:
#     execute_sql('''SET query_band='DEMO=Data_Loading_and_Descriptive_Statistics.ipynb;' UPDATE FOR SESSION; ''')


# Begin running steps with `Shift + Enter` keys.

# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## Python Reusable Functions

# In[8]:


## Define Teradata Vantage: schema & table name
SCHEMA_NAME= 'Air_Pollution'
TABLE_NAME = f'Air_Pollution_{USE_DATA_PREFIX}'

## Define the mapping of original column names to new column names
columns_name_mapping = {
    'Timestamp (UTC+12:00)':'Timestamp',
    'AQI.Air Quality Index (AQI)':'AQI',
    'PM10.Hourly Aggregate (µg/m³)':'PM10',
    'PM2,5.Hourly Aggregate (µg/m³)':'PM2.5',
    'NO.Hourly Aggregate (µg/m³)':'NO',
    'NO2.Hourly Aggregate (µg/m³)':'NO2',
    'NOx.Hourly Aggregate (µg/m³)':'NOx',
    'CO.Hourly Aggregate (mg/m³)':'CO',                 ## No Data in both Site1 (Penrose) and Site2 (Takapuna)
    'O3.Hourly Aggregate (µg/m³)':'O3',                 ## No Data in both Site1 (Penrose) and Site2 (Takapuna)
    'SO2.Hourly Aggregate (µg/m³)':'SO2',               ## No Data in Site2 (Takapuna)
    'Wind Speed.Hourly Aggregate (m/s)':'Wind_Speed',
    'Wind Dir.Hourly Aggregate (°)':'Wind_Dir',
    'Air Temp.Hourly Aggregate (°C)':'Air_Temp',
    'Rel Humidity.Hourly Aggregate (%)':'Rel_Humidity',
    # 'Solar Rad.Hourly Aggregate (W/m²)':'Solar_Rad',  ## No Data in both Site1 (Penrose) and Site2 (Takapuna)
}

# columns_name_mapping_cleaned = {
#     'Timestamp':'Timestamp',
#     'AQI':'AQI',
#     'PM10':'PM10',
#     'PM2.5':'PM2.5',
#     'SO2':'SO2',                                        ## No Data in Site2
#     'NO':'NO',
#     'NO2':'NO2',
#     'NOx':'NOx',
#     'Wind_Speed':'Wind_Speed',
#     'Wind_Dir':'Wind_Dir',
#     'Air_Temp':'Air_Temp',
#     'Rel_Humidity':'Rel_Humidity',
# }


# In[33]:


### [DEBUG] Listing of all input data files under "input/" directory: *.csv, *.xlsx
import os

print(f'\nℹ️ Load Data from {DATA_PATH} folder\n')
for dirname, _, filenames in os.walk(DATA_PATH):
    for filename in filenames:
        ## Check if the filename ends with .csv or .xlsx
        if filename.endswith('.csv') or filename.endswith('.xlsx'):
            logging.info(os.path.join(dirname, filename))
            # pass


# <header style="padding:3px;border-top:1px solid black">
# 
# ### Load Data from *.csv files

# In[10]:


def ensure_datetime_format(df, column_name='Timestamp'):
    """
    Ensures the specified column is in datetime format.
    Parameters:
    - df (pd.DataFrame): The dataframe to process.
    - column_name (str): The name of the column to format as datetime.
    Returns:
    - pd.DataFrame: DataFrame with the column in datetime format.
    """
    df_datetime_column = df[column_name].copy()
    try:
        df_datetime_column = pd.to_datetime(df_datetime_column, dayfirst=True, errors='coerce')
        # df_datetime_column =df_datetime_column.tz_localize("NZ")
        # df_datetime_column = pd.to_datetime(df_datetime_column, format='%d/%m/%Y %H:%M')
    except Exception as e:
        raise ValueError(f"Error converting {column_name} to datetime: {e}")
        
    return df_datetime_column


# In[11]:


def load_data_from_file(file_path, site_name, site_class, include_columns=[]):

    ## Read the dataset from the CSV file
    logging.info(f"\nℹ️ Load Data from {file_path} file --> rawdata DataFrame 📂")
    raw_data = pd.read_csv(file_path)
    # raw_data = pd.read_excel(file_path)

    if USE_DATA_PREFIX.lower() == 'raw':    
        ## Rename the columns in the DataFrame
        raw_data = raw_data.rename(columns=columns_name_mapping)
    # else:
    #     raw_data = raw_data.rename(columns=columns_name_mapping_cleaned)

    ## Dynamically select columns specified in columns_name_mapping that are not in exclude_columns and are present in the DataFrame
    # selected_columns = [col for col in columns_name_mapping.values() if col not in exclude_columns and col in raw_data.columns]
    raw_data = raw_data[include_columns]
    
    ## Add site name and site class columns
    raw_data['Site'] = site_name
    # raw_data['Site_Class'] = site_class
    # raw_data['Country'] = 'New Zealand'  ## TODO: Hard-code

    ## Convert 'Timestamp' column to datetime ("29/12/2020 00:00:00" as December 29, 2020) & addressing the date format warning.
    # raw_data['Timestamp'] = pd.to_datetime(raw_data['Timestamp'], dayfirst=True)
    raw_data['Timestamp'] = ensure_datetime_format(raw_data, 'Timestamp')
    
    return raw_data


# <header style="padding:3px;border-top:1px solid black">
# 
# ### Prepare Data in Teradata Vantage

# <div class="alert alert-block alert-info">
# 🎓 Using `fastload()` for high performance data loading of large amounts of data into a table on Teradata Vantage.
# </div>
# 
# * The `fastload()` API writes records from a `Pandas DataFrame` to `Teradata Vantage` using Fastload. FastLoad API can be used to quickly load large amounts of data in an empty table on Vantage.
# 
# * `Fastload` protocol is excellent for row counts over 100K - shown here as an illustration. These Teradata functions have lots of parameters to help control behaviour - the if_exists parameter is excellent, so we don't have to explicitly drop the table before loading it - or we can append it, etc. We can also use copy_to_sql for smaller row counts and more flexibility.

# In[12]:


# def load_and_prepare_data_in_database(raw_data, schema_name, table_name, if_exists='replace'):
def load_and_prepare_data_in_database(raw_data, table_name, if_exists='replace'):
    """
    Loads data from a Pandas DataFrame into a Teradata table using teradataml,
    and prepares the data by performing specified SQL transformations.

    Parameters:
    - df: Pandas DataFrame containing the data to load.
    - table_name: The name of the target table in Teradata.
    - if_exists: Action to take if the target table already exists. Options include 'replace'.
    """
    
    ## [DEBUG] Little bit of code that creates an index
    # raw_data['txn_id'] = range(1, len(raw_data) + 1)
    ## FIXME: 'Site' & 'Timestamp'

    # copy_to_sql(raw_data, table_name=TABLE_NAME, if_exists="replace")
    fastload(raw_data,
             # schema_name=schema_name,
             table_name=table_name,
             index=False,
             # primary_index='txn_id',
             primary_index=['Site', 'Timestamp'],
             if_exists = if_exists,
             open_sessions=2)
    
    # return DataFrame.from_table(table_name)
    return TeradataDataFrame(table_name)


# <header style="padding:3px;border-top:1px solid black">
# 
# ### Descriptive Statistics

# In[13]:


class DescriptiveStatistics:

    @staticmethod
    def describe_data(_data, ordinal_columns, numerical_columns=None):

        ## The Shape | Head (First 5 rows) | Tail (Last 5 rows) of the Dataframe
        # nrow, ncol = _data.shape
        # print("\nℹ️ The Shape of the Dataframe: \n", _data.shape)
        # print(nrow, ncol)

        ## Sums up the number of True values from missing values are contained within the dataframe.
        # missing_data = _data.isnull().sum()
        # print("\nℹ️ Sums up missing values: \n", missing_data)

        ## Summary Statistics of the Dataframe such as the mean, maximum and minimum values
        # print("\nℹ️ First 5 rows of the Dataframe: \n", _data.head())
        # print("\nℹ️ Last 5 rows of the Dataframe: \n", _data.tail())
        # print("\nℹ️ Information about the dataframe and count of the non-null values: \n", _data.info())
        # print("\nℹ️ Summary Statistics  (count | mean | std | min | max) of the dataframe: \n", _data.describe())

        # print("\n🎓 Describing the types of each attribute as Nominal, Ordinal, or Continuous ...")
       
        ## Data-type of each columns directly from their data types except those listed in ordinal_columns
        # print("\nℹ️ Data-type of each column in the dataframe: \n", _data.dtypes)
        # numerical_columns = _data.select_dtypes(include=[np.number]).columns
        numerical_columns = [col for col in _data.select_dtypes(include=[np.number]).columns if col not in ordinal_columns]

        ## Identifying nominal (categorical) columns; these are typically of object or category dtype
        ## Note: This includes 'Site' and 'Site_Class', and any other categorical columns present
        nominal_columns = _data.select_dtypes(include=[object, 'category']).columns
        # nominal_columns = _data.select_dtypes(include=['object', 'category']).columns.tolist()
        ## Ensuring 'Timestamp' is not mistakenly classified as nominal
        numerical_columns = [col for col in numerical_columns if col not in ordinal_columns]

        # Print the classification of columns
        print("\nℹ️ Numerical Variables/Features: \n", numerical_columns)
        print("\nℹ️ Ordinal Variables/Features: \n", ordinal_columns)
        if nominal_columns is not None and len(nominal_columns) > 0:
            print("\nℹ️ Nominal Variables/Features: \n", nominal_columns)

        ## Return the lists for further use
        return numerical_columns, nominal_columns

    @staticmethod
    def check_duplicate_rows(_data):
        """
        Verify_data_quality
        """
        duplicates = _data.duplicated()
        print("\nℹ️ Number of duplicate rows: ", duplicates.sum())
        if duplicates.sum() > 0:
            print("\nDuplicate rows: \n", _data[duplicates])
        return duplicates


    @staticmethod
    def correlations_heatmap_with_regression(data, continuous_columns, hue_column='Site', corner=False):
        """
        This function takes a DataFrame and a list of continuous column names and creates a seaborn seaborn's PairGrid 
        with regression lines for continuous columns, colored by hue_column, in the lower triangle,
        KDE plots in the upper, and histograms in the diagonal.
        
        Parameters:
        - data (DataFrame): The data to be plotted.
        - continuous_columns (list): List of continuous column names for the pairplot.
        - hue_column: Column name to use for hue differentiation. Default is 'Site'.
        - corner: If True, only plots the lower triangle of the pair grid to avoid redundancy. Default is False.
        """

        ## Check if hue_column exists: 'Site'
        if hue_column not in data.columns:
            raise KeyError(f"'{hue_column}' column not found in the data.")
        
        ## [DEBUG] Resetting the index so 'Site' becomes a column again, suitable for use as a hue in pairplot
        ## Ensure 'Site' is a column in the DataFrame for hue mapping
        # df = data.reset_index() if 'Site' not in data.columns else data
        # df = data.reset_index()
        # Replace inf/-inf with NaN to avoid plotting issues
        df = data.replace([np.inf, -np.inf], np.nan)

        # ## Determine the number of unique 'Site' values
        # num_unique_sites = _df['Site'].nunique()    
        # ## Generate a list of markers, one for each unique 'Site' value: ['o', 's', 'D', '^', ...] up to the number of unique sites
        # marker_list = ['o', 's', 'D', '^', '<', '>', 'v', '*', '+', 'x'][:num_unique_sites]
        
        ## Define PairGrid
        # pair_grid_kwargs = {'hue': hue_column, 'palette': "Set2", 'diag_sharey': False}
        if corner:
            # pair_grid = sns.PairGrid(df[continuous_columns + [hue_column]], corner=True, **pair_grid_kwargs)
            pair_grid = sns.PairGrid(df, hue="Site", palette="Set2", diag_sharey=False, hue_kws={"marker": ['o', 's']}, corner=True)
        else:
            # pair_grid = sns.PairGrid(df[continuous_columns + [hue_column]], **pair_grid_kwargs)
            pair_grid = sns.PairGrid(df, hue="Site", palette="Set2", diag_sharey=False, hue_kws={"marker": ['o', 's']})
            ## KDE plot for the upper triangle
            pair_grid.map_upper(sns.kdeplot, shade=True)
        
        ## Histogram for the diagonal
        pair_grid.map_diag(sns.histplot, alpha=0.5)  ## sns.kdeplot
        ## [Left] Regression plot for the lower triangle
        pair_grid.map_lower(sns.regplot) ## sns.scatterplot
        
        ## Add legend and adjust title
        pair_grid.add_legend(loc='upper left', ncol=3)
        plt.suptitle("Pairplot with Regression Lines by " + hue_column, y=1.02, size='x-large')
        ## Set the title of the figure
        # pair_grid.fig.suptitle("Pairplot with Regression Lines by Site", y=1.01, size='x-large')  ## Adjust title and its position
        
        ## Display the plot
        plt.show()

    @staticmethod
    def correlations_matrix(df, method='pearson', figsize=(20, 10), title=None, _rotate=False):
        """
        Generates a heatmap for the correlation matrix of a DataFrame.
        
        Parameters:
        - df (DataFrame): Pandas DataFrame containing only numerical columns.
        - method (str): Correlation computation method, 'pearson' (default), 'spearman', or 'kendall'.
        - figsize (tuple): Figure size for the heatmap.
        - title (str): Title for the heatmap.
        
        Returns:
        - _correlation_matrix: The function displays a heatmap.
        """
    
        if method not in ['pearson', 'spearman', 'kendall']:   ## Validate correlation method
            raise ValueError("Method should be 'pearson', 'spearman', or 'kendall'.")
    
        ## Make a copy of the df
        data = df.copy()
        ## Check features correlation
        _correlation_matrix = df.corr(method=method)
        
        ## Create a mask to hide the upper triangle
        # mask = np.triu(np.ones_like(_correlation_matrix, dtype=bool))
        mask = np.zeros_like(_correlation_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        ## Set the diagonal elements of the mask to False to display self-correlation
        np.fill_diagonal(mask, False)
        
        fig, ax = plt.subplots(figsize=figsize)
        plt.title(title, fontsize='x-large', y=1.01)
        
        ## Visualize the Correlation Matrix by plotting a Heatmap with the custom mask
        plt.rcParams.update({'font.size': 12})
        heatmap = sns.heatmap(_correlation_matrix,
                              annot=True, annot_kws={"fontsize": 10}, ## Adjust annotation font size
                              fmt=".2f", linewidths=0.5, cmap='RdBu', ## 'coolwarm' 'viridis' 'RdBu'
                              square=True, 
                              # cbar_kws={"shrink": .5},
                              cbar_kws = {'shrink': .4,"ticks" : [-1, -.5, 0, 0.5, 1]},
                              vmin = -1, vmax = 1,
                              mask=mask,                              ## the mask has been included here
                              ax=ax)
         
        ## Display the plot
        if _rotate:
            ## Invert the y-axis to "rotate" the heatmap
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
     
        ## Additionally, rotate the xticks if it improves readability
        plt.yticks(rotation=90)
        # plt.tight_layout()
        plt.show()
    
        return _correlation_matrix

# print(plt.style.available)
# plt.style.use('tableau-colorblind10')


# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # 📂 Data Loading

# <div class="alert alert-block alert-info">
# 🎓 Load Data from *.csv files and/or Prepare Data in Teradata Vantage
# </div>

# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Load and Prepare Data from *.csv files

# In[16]:


def load_and_prepare_data_from_files(data_path, use_data_prefix, sites_configuration):
    """
    Configures and loads data for multiple sites based on the specified configurations.
    
    Parameters:
    - data_path (str): Base path for the data files.
    - use_data_prefix (str): Prefix to append before file names.
    - sites_configuration (list): A list of dictionaries containing site data configurations.
    
    Returns:
    - DataFrame: A concatenated DataFrame of all sites' data.
    """
    # global numerical_columns, include_columns_site1, include_columns_site2 
    # numerical_columns     = ['Timestamp','AQI','PM10','PM2.5','SO2','NO','NO2','NOx','Wind_Speed','Wind_Dir','Air_Temp','Rel_Humidity']
    # include_columns_site1 = ['Timestamp','AQI','PM10','PM2.5','SO2','NO','NO2','NOx','Wind_Speed','Wind_Dir','Air_Temp','Rel_Humidity']
    # include_columns_site2 = ['Timestamp','AQI','PM10','PM2.5','NO','NO2','NOx','Wind_Speed','Wind_Dir','Air_Temp','Rel_Humidity']
    # 
    # if USE_DATA_PREFIX.lower() == 'raw':    
    #     ## Define the specifications for each monitoring site
    #     sites_info = [
    #         {"path": f"{DATA_PATH}/Penrose7-07May2020-to-30Apr2022.csv", "name": "Penrose", "class": "Industrial / Traffic", "include_columns": include_columns_site1},
    #         {"path": f"{DATA_PATH}/Takapuna23-07May2020-to-30Apr2022.csv", "name": "Takapuna", "class": "Urban Background", "include_columns": include_columns_site2},
    #         # {"path": f"{DATA_PATH}/air-pm-QueenStreet9-30Dec2019-to-29Feb2024.csv", "name": "QueenStreet", "class": "Traffic Peak", "exclude_columns": ['CO', 'O3', 'Wind_Speed', 'Wind_Dir', 'Air_Temp', 'Rel_Humidity', 'Solar_Rad']}
    #     ]
    # else:
    #     ## Define the specifications for each monitoring site
    #     sites_info = [
    #         {"path": f"{DATA_PATH}/{USE_DATA_PREFIX}_Penrose7-07May2020-to-30Apr2022.csv", "name": "Penrose", "class": "Industrial / Traffic", "include_columns": include_columns_site1},
    #         {"path": f"{DATA_PATH}/{USE_DATA_PREFIX}_Takapuna23-07May2020-to-30Apr2022.csv", "name": "Takapuna", "class": "Urban Background", "include_columns": include_columns_site2},
    #         # {"path": f"{DATA_PATH}/air-pm-QueenStreet9-30Dec2019-to-29Feb2024.csv", "name": "QueenStreet", "class": "Traffic Peak", "exclude_columns": ['CO', 'O3', 'Wind_Speed', 'Wind_Dir', 'Air_Temp', 'Rel_Humidity', 'Solar_Rad']}
    #     ]
    # 
    ## Define the specifications for each monitoring site
    # sites_info = [
    #     {"path": f"{DATA_PATH}/{USE_DATA_PREFIX}_Penrose7-07May2020-to-30Apr2022.csv", "name": "Penrose", "class": "Industrial / Traffic", "include_columns": include_columns_site1},
    #     {"path": f"{DATA_PATH}/{USE_DATA_PREFIX}_Takapuna23-07May2020-to-30Apr2022.csv", "name": "Takapuna", "class": "Urban Background", "include_columns": include_columns_site2},
    #     # {"path": f"{PATH}/{USE_DATA_PREFIX}_QueenStreet9-30Dec2019-to-29Feb2024.csv", "name": "QueenStreet", "class": "Traffic Peak", "exclude_columns": ['CO', 'O3', 'Wind_Speed', 'Wind_Dir', 'Air_Temp', 'Rel_Humidity', 'Solar_Rad']}
    # ]
    # 
    ## Load and prepare the data for each site
    # dfs = [load_data_from_file(site['path'], site['name'], site['class'], site['include_columns']) for site in sites_info]
    # 
    ## Concatenate all the dataframes into one
    # data = pd.concat(dfs, ignore_index=True)
    
    dfs = []
    for site in sites_configuration:
        file_path = f"{data_path}/{use_data_prefix}_{site['file_suffix']}"
        df = load_data_from_file(file_path, site['name'], site['class'], site['include_columns'])
        if not df.empty:
            dfs.append(df)

    if dfs:
        data = pd.concat(dfs, ignore_index=True)
        # data.head().style.set_properties(subset=['Timestamp'], **{'background-color': 'dodgerblue'})
        print("\nℹ️ Combined Data Shape:", data.shape)
        return data
    else:
        logging.info("No data loaded for any site.")
        return pd.DataFrame()


# In[23]:


if IS_LOADING_FROM_FILES and IS_TEST_DEV:
    ## Monitoring Sites Configuration
    global numerical_columns, include_columns_site1, include_columns_site2 
    numerical_columns     = ['Timestamp','AQI','PM10','PM2.5','SO2','NO','NO2','NOx','Wind_Speed','Wind_Dir','Air_Temp','Rel_Humidity']
    include_columns_site1 = ['Timestamp','AQI','PM10','PM2.5','SO2','NO','NO2','NOx','Wind_Speed','Wind_Dir','Air_Temp','Rel_Humidity']
    include_columns_site2 = ['Timestamp','AQI','PM10','PM2.5','NO','NO2','NOx','Wind_Speed','Wind_Dir','Air_Temp','Rel_Humidity']

    sites_configuration = [
        {"file_suffix": "Penrose7-07May2020-to-30Apr2022.csv", "name": "Penrose", "class": "Industrial / Traffic", "include_columns": include_columns_site1},
        {"file_suffix": "Takapuna23-07May2020-to-30Apr2022.csv", "name": "Takapuna", "class": "Urban Background", "include_columns": include_columns_site2}
    ]
    rawdata = load_and_prepare_data_from_files(data_path=DATA_PATH, use_data_prefix=USE_DATA_PREFIX, sites_configuration=sites_configuration)

    # numerical_columns     = ['Timestamp','AQI','PM10','PM2.5','SO2','NO','NO2','NOx','Wind_Speed','Wind_Dir','Air_Temp','Rel_Humidity']
    # include_columns_site1 = ['Timestamp','AQI','PM10','PM2.5','SO2','NO','NO2','NOx','Wind_Speed','Wind_Dir','Air_Temp','Rel_Humidity']
    # include_columns_site2 = ['Timestamp','AQI','PM10','PM2.5','NO','NO2','NOx','Wind_Speed','Wind_Dir','Air_Temp','Rel_Humidity']

    # # if USE_DATA_PREFIX.lower() == 'raw':    
    # #     ## Define the specifications for each monitoring site
    # #     sites_info = [
    # #         {"path": f"{DATA_PATH}/Penrose7-07May2020-to-30Apr2022.csv", "name": "Penrose", "class": "Industrial / Traffic", "include_columns": include_columns_site1},
    # #         {"path": f"{DATA_PATH}/Takapuna23-07May2020-to-30Apr2022.csv", "name": "Takapuna", "class": "Urban Background", "include_columns": include_columns_site2},
    # #         # {"path": f"{DATA_PATH}/air-pm-QueenStreet9-30Dec2019-to-29Feb2024.csv", "name": "QueenStreet", "class": "Traffic Peak", "exclude_columns": ['CO', 'O3', 'Wind_Speed', 'Wind_Dir', 'Air_Temp', 'Rel_Humidity', 'Solar_Rad']}
    # #     ]
    # # else:
    # #     ## Define the specifications for each monitoring site
    # #     sites_info = [
    # #         {"path": f"{DATA_PATH}/{USE_DATA_PREFIX}_Penrose7-07May2020-to-30Apr2022.csv", "name": "Penrose", "class": "Industrial / Traffic", "include_columns": include_columns_site1},
    # #         {"path": f"{DATA_PATH}/{USE_DATA_PREFIX}_Takapuna23-07May2020-to-30Apr2022.csv", "name": "Takapuna", "class": "Urban Background", "include_columns": include_columns_site2},
    # #         # {"path": f"{DATA_PATH}/air-pm-QueenStreet9-30Dec2019-to-29Feb2024.csv", "name": "QueenStreet", "class": "Traffic Peak", "exclude_columns": ['CO', 'O3', 'Wind_Speed', 'Wind_Dir', 'Air_Temp', 'Rel_Humidity', 'Solar_Rad']}
    # #     ]
    
    # ## Define the specifications for each monitoring site
    # sites_info = [
    #     {"path": f"{DATA_PATH}/{USE_DATA_PREFIX}_Penrose7-07May2020-to-30Apr2022.csv", "name": "Penrose", "class": "Industrial / Traffic", "include_columns": include_columns_site1},
    #     {"path": f"{DATA_PATH}/{USE_DATA_PREFIX}_Takapuna23-07May2020-to-30Apr2022.csv", "name": "Takapuna", "class": "Urban Background", "include_columns": include_columns_site2},
    #     # {"path": f"{PATH}/{USE_DATA_PREFIX}_QueenStreet9-30Dec2019-to-29Feb2024.csv", "name": "QueenStreet", "class": "Traffic Peak", "exclude_columns": ['CO', 'O3', 'Wind_Speed', 'Wind_Dir', 'Air_Temp', 'Rel_Humidity', 'Solar_Rad']}
    # ]
        
    
    # ## Load and prepare the data for each site
    # dfs = [load_data_from_file(site['path'], site['name'], site['class'], site['include_columns']) for site in sites_info]
    
    # ## Concatenate all the dataframes into one
    # rawdata = pd.concat(dfs, ignore_index=True)
    
    # ## Display the first few rows of the final dataframe
    # # rawdata.head().style.set_properties(subset=['Timestamp'], **{'background-color': 'dodgerblue'})
    # print("\nℹ️ The Shape of the rawdata Dataframe:", rawdata.shape)


# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Prepare Data in Teradata Vantage

# In[24]:


## Code to execute if data is in Teradata Vantage and it's the ClearScape Analytics™ environment
if IS_DATA_IN_TERADATA_VANTAGE and IS_LOADING_FROM_FILES and IS_LOADING_FROM_FILES:

    print("\nℹ️ Prepare Data in Teradata Vantage from Pandas DataFrame \n")
    if_exists = 'replace'
    
    # df_table = load_and_prepare_data_in_database(rawdata, SCHEMA_NAME, TABLE_NAME, if_exists=if_exists)
    df_table = load_and_prepare_data_in_database(rawdata, TABLE_NAME, if_exists=if_exists)
    
    ## [DEBUG] Info
    print(f'\nℹ️ Loading Data in Teradata Vantage: {type(df_table)}; df_table.shape: {df_table.shape}')


# In[25]:


## Code to execute if data is in Teradata Vantage and it's the ClearScape Analytics™ environment
if IS_DATA_IN_TERADATA_VANTAGE and IS_TERADATA_VANTAGE and not IS_LOADING_FROM_FILES:
    
    print("\nℹ️ Loading Data in Teradata Vantage to Pandas DataFrame \n")
    df_table = TeradataDataFrame(TABLE_NAME, index_label=["Site", "Timestamp"])
    # df_table = TeradataDataFrame.from_table(f'"{SCHEMA_NAME}"."{TABLE_NAME}"')
    
    # df_table
    # df_table.shape
    # db_list_tables(object_name = "%")
    
    rawdata = df_table.to_pandas()


# In[26]:


# rawdata


# <header style="padding:3px;border-top:1px solid #E37C4D">
# 
# ### Cleanup Vantage

# In[27]:


IS_CLEANUP_VANTAGE=False

if IS_DATA_IN_TERADATA_VANTAGE:
    db_list_tables(object_name = "%")

## Code to execute if data is in Teradata Vantage and it's the ClearScape Analytics™ environment
if IS_DATA_IN_TERADATA_VANTAGE and IS_CLEANUP_VANTAGE:
    
    tables = [TABLE_NAME]
    
    ## Loop through the list of tables and execute the drop table command for each table
    for table in tables:
        try:
            print(f"To drop the table {TABLE_NAME}, uncomment the following line of code")
            db_drop_table(table_name=table)
        except:
            pass

    print(f"To remove the context, uncomment the following line of code")
    remove_context()


# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # Descriptive Statistics

# In[34]:


if IS_TEST_DEV and IS_JUPYTERLAB:
    ## The rawdata is retrieved from "2.1. Load Data from *.csv files" and/or "2.2. Prepare Data in Teradata Vantage"
    rawdata_site1 = rawdata[rawdata['Site'] == 'Penrose'][include_columns_site1]
    rawdata_site2 = rawdata[rawdata['Site'] == 'Takapuna'][include_columns_site2]
    print("ℹ️ The Shape of the Dataframe rawdata_site1 (Penrose) and rawdata_site2 (Takapuna):", rawdata_site1.shape, rawdata_site2.shape)
    # logging.info("ℹ️ The Shape of the DataFrame rawdata_site1 (Penrose) and rawdata_site2 (Takapuna): %s, %s", rawdata_site1.shape, rawdata_site2.shape)


# In[35]:


if IS_TEST_DEV and IS_JUPYTERLAB:

    ## 1. Describe Data and attribute-types: numerical(continuous), categorical (nominal, ordinal) 
    
    ## 'Timestamp' as the only Ordinal Attribute/Column given its nature order in time-series data
    ordinal_columns = ['Timestamp']
    
    print("\n🎓 Describing the types of each attribute as numerical_columns (Continuous), ordinal_columns (Ordinal), or nominal_columns (Nominal) ...")
    numerical_columns, nominal_columns = DescriptiveStatistics.describe_data(rawdata, ordinal_columns, numerical_columns)
    # print("\n🎓 0. Summary Statistics of the Dataframe such as the mean, maximum and minimum values ...")
    # rawdata.describe().T.style.set_table_attributes("style='display:inline'").bar(color='#3dc4c4') ## color='dodgerblue'
    # rawdata.describe()
    
    print("\n🎓 1. [Site 1 - Penrose][numerical_columns_S1, nominal_columns_S1] Summary Statistics of the Dataframe such as the mean, maximum and minimum values ...")
    numerical_columns_site1, nominal_columns_site1 = DescriptiveStatistics.describe_data(rawdata_site1, ordinal_columns, include_columns_site1)
    # rawdata_site1.describe().T.style.set_table_attributes("style='display:inline'").bar(color='dodgerblue') ## color='dodgerblue' | '#3dc4c4'
    
    print("\n🎓 2. [Site 2 - Takapuna][numerical_columns_S2, nominal_columns_S2]  Summary Statistics of the {site2} Dataframe such as the mean, maximum and minimum values ...")
    numerical_columns_site2, nominal_columns_site2 = DescriptiveStatistics.describe_data(rawdata_site2, ordinal_columns, include_columns_site2)
    
    ## Verify Data Quality: check_duplicate_rows and check_missing_values
    # duplicate_records = DescriptiveStatistics.check_duplicate_rows(rawdata)


# In[36]:


if IS_TEST_DEV:
    print("\n🎓 [Site1 - Penrose]  Summary Statistics of the {site1} rawdata_site1 Dataframe such as the mean, max/minimum values ...")
    # rawdata_site1.describe()


# In[37]:


if IS_TEST_DEV:
    print("\n🎓 [Site2 - Takapuna]  Summary Statistics of the {site2} rawdata_site2 Dataframe such as the mean, max/minimum values ...")
    # rawdata_site2.describe()


# In[39]:


if IS_TEST_DEV:
    ## 2. Correlations Heatmap with Regression
    DescriptiveStatistics.correlations_heatmap_with_regression(rawdata, numerical_columns, 'Site', corner=True)
    DescriptiveStatistics.correlations_heatmap_with_regression(rawdata, numerical_columns, 'Site', corner=False)


# In[40]:


## 3. Correlations Heatmap 
if IS_TEST_DEV:
    ## Filter rawdata for the specified sites
    ## Copy the rawdata for safety so the original data is not modified
    ## Set 'Site' and 'Timestamp' as a multi-level index
    # rawdata.set_index(['Site', 'Timestamp'], inplace=True)
    print("\n🎓 If data is skewed, transform the data (using statsmodels/pmdarima's BoxCoxEndogTransformer or LogEndogTransformer transformation) to stabilise the variance.")
    # rawdata_site1 = rawdata.xs('Penrose', level='Site')[numerical_columns_S1]
    # numerical_columns_2 = numerical_columns.drop('SO2') ## Removing 'SO2' from the Index
    # rawdata_site2 = rawdata.xs('Takapuna', level='Site')[numerical_columns_S1]
    
    correlation_matrix1 = DescriptiveStatistics.correlations_matrix(rawdata_site1, method='pearson', figsize=(20,15), title="[Site1 - Penrose] Correlation Matrix Heatmap")
    
    correlation_matrix2 = DescriptiveStatistics.correlations_matrix(rawdata_site2, method='pearson', figsize=(20,15), title="[Site2 - Takapuna] Correlation Matrix Heatmap",_rotate=True)


# In[41]:


if IS_TEST_DEV:
    print("\n🎓 [Site1 - Penrose] Correlation Matrix Heatmap:")
    correlation_matrix1


# In[42]:


if IS_TEST_DEV:
    print("\n🎓 [Site2 - Takapuna] Correlation Matrix Heatmap:")
    correlation_matrix2


# In[ ]:


# from prettytable import PrettyTable

def display_site_comparison():
    """
    This function creates and displays a table that lists various data attributes along with descriptions and 
    their applicability to different datasets across sites using the PrettyTable library.
    """
    
    ## Define the PrettyTable table columns and set column alignments
    table = PrettyTable()
    table.field_names = ["Variable Name", "Description", "All Sites", "Penrose", "Takapuna"]
    table.align["Variable Name"] = "l"
    table.align["Description"] = "l"
    table.align["All Sites"] = "c"
    table.align["Penrose"] = "c"
    table.align["Takapuna"] = "c"

    ## Define the data dictionary with clear structure that containing information about the datasets and placeholders for potential future expansion
    data_dictionary = [
        ("rawdata", "Complete dataset containing all observations across all sites.", "[x]", "[x]", "[x]"),
        ("ordinal_columns", "Ordinal columns specific to Site 1.", "[x]", "[x]", "[x]"),
        ("numerical_columns_site1", "Numerical columns specific to Site 1.", "[ ]", "[x]", "[ ]"),
        ("nominal_columns_site1", "Nominal columns specific to Site 1.", "[ ]", "[x]", "[ ]"),
        ("numerical_columns_site2", "Numerical columns specific to Site 2.", "[ ]", "[ ]", "[x]"),
        ("nominal_columns_site2", "Nominal columns specific to Site 2.", "[ ]", "[ ]", "[x]"),
        ("rawdata_site1", "Subset of raw data for Site 1.", "[ ]", "[x]", "[ ]"),
        ("rawdata_site2", "Subset of raw data for Site 2.", "[ ]", "[ ]", "[x]"),
        # ("---------------------------", "---------------------------------------------------------------------", "---------", "-------", "--------"),  ## Blank line for separation
        # ("cleaned_data", "Cleaned dataset with preprocessing applied.", "[x]", "[x]", "[x]"),
        # ("cleaned_ordinal_columns", "Ordinal columns in the cleaned dataset.", "[x]", "[x]", "[x]"),
        # ("cleaned_numerical_columns", "Numerical columns in the cleaned dataset.", "[x]", "[x]", "[x]"),
        # ("cleaned_nominal_columns", "Nominal columns in the cleaned dataset.", "[x]", "[x]", "[x]"),
        # ("cleaned_data1", "Cleaned data for Site 1.", "[ ]", "[x]", "[ ]"),
        # ("cleaned_data2", "Cleaned data for Site 2.", "[ ]", "[ ]", "[x]"),
        # ("---------------------------", "---------------------------------------------------------------------", "---------", "-------", "--------"),  ## Blank line for separation
        # ("top_features_PM25_site1", "Top features correlated with PM2.5 at Site 1 (Penrose).", "[ ]", "[x]", "[ ]"),
        # ("top_features_PM25_site2", "Top features correlated with PM2.5 at Site 2 (Takapuna).", "[ ]", "[ ]", "[x]"),
        # ("top_features_PM10_site1", "Top features correlated with PM10 at Site 1.", "[ ]", "[x]", "[ ]"),
        # ("top_features_PM10_site2", "Top features correlated with PM10 at Site 2.", "[ ]", "[ ]", "[x]"),
        # ("summary_stats_PM25_penrose", "[PM2.5] Summary statistics for the Penrose site after preprocessing.", "[ ]", "[x]", "[ ]"),
        # ("summary_stats_PM25_takapuna", "[PM2.5] Summary statistics for the Takapuna site after preprocessing.", "[ ]", "[ ]", "[x]"),
        # ("summary_stats_PM10_penrose", "[PM10] Summary statistics for the Penrose site after preprocessing.", "[ ]", "[x]", "[ ]"),
        # ("summary_stats_PM10_takapuna", "[PM10] Summary statistics for the Takapuna site after preprocessing.", "[ ]", "[ ]", "[x]"),
        # ("selected_features_PM25_penrose", "[PM2.5] Target + Selected Features for the Penrose site.", "[ ]", "[x]", "[ ]"),
        # ("selected_features_PM25_takapuna", "[PM2.5] Target + Selected Features for the Takapuna site.", "[ ]", "[ ]", "[x]"),
        # ("selected_features_PM10_penrose", "[PM10] Target + Selected Features for the Penrose site.", "[ ]", "[x]", "[ ]"),
        # ("selected_features_PM10_takapuna", "[PM10] Target + Selected Features for the Takapuna site.", "[ ]", "[ ]", "[x]"),
    ]

    ## Format for better readability
    for entry in data_dictionary:
        ## Adding rows to the table
        table.add_row(entry)

    ## Print the table in an organized format
    print(table)


if IS_TEST_DEV and IS_JUPYTERLAB:    
    ## Call the function to display the table
    print('\n🎓 [Data_Loading_and_Descriptive_Statistics.ipynb] Listing variables with description...')
    display_site_comparison()


# <div class="alert alert-block alert-success">
# 
# 🚀 <a href="https://analytics-experience.pages.dev/html/Data_Loading_and_Descriptive_Statistics" target="_blank">Project Documentation at https://analytics-experience.pages.dev</a>
#     
# 🎓 Next Steps: 
#     
# 1. [x] [Agile CRISP-DM >> Data Understanding](./1_Data_Understanding.ipynb)
# 2. [x] [Agile CRISP-DM >> Data Preparation](./2_Data_Preparation.ipynb)
# 
# </div>

# <footer style="padding-bottom:35px; background:#f9f9f9; border-bottom:3px solid #00b2b1">
#     <div style="float:left;margin-top:14px;color:#E37C4D">Predicting Air Particulate Matter at Scale ⛅️</div>
#     <div style="float:right;">
#         <div style="float:left; margin-top:14px">
#             Auckland University of Technology (AUT) 🎓
#         </div>
#     </div>
# </footer>
