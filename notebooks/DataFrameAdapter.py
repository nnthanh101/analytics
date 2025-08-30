#!/usr/bin/env python
# coding: utf-8

# <header style="padding:1px;background:#f9f9f9;border-top:3px solid #00b2b1"><img id="Teradata-logo" src="https://www.teradata.com/Teradata/Images/Rebrand/Teradata_logo-two_color.png" alt="Teradata" width="220" align="right" />
# 
# <b style = 'font-size:28px;font-family:Arial;color:#E37C4D'>🎓 Predicting Air Particulate Matter at Scale ⛅️</br> 🛠️ [Reusable] Final Data & Features</b>
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
# 🎓 This reusable notebook shows the cleaned Time Series Dataset and relevant features. It should be ready for reuse in the next steps (Machine Learning / Deep Learning Modelling and Intelligent Dashboard) in CRISP-DM for Data Science.
#     
# </div>

# * **Data:**
# 
#   * [x] Six key air pollutants from WHO’s standard, including PM10 and PM2.5 (particulate matter with diameters of 10 and 2.5 microns), ozone (O3), nitrogen dioxide (NO2), sulphur dioxide (SO2), and carbon monoxide (CO).
#   * [x] Meteorological/Weather factors: wind speed, wind direction, air temperature, and relative humidity are crucial to determining the spread and behaviour of PM2.5/PM10 pollutants.
#   * [x] Features: 
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
# 🎓 This section executes all of the cells in `Data_Loading_and_Descriptive_Statistics.ipynb`.
# </div>

# In[1]:


import os, logging
import pandas as pd              ## Data processing, file I/O
import numpy as np               ## Linear algebra

## Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## .env --> Setting the environment variable for output format programmatically
# os.environ['ENV_PATH'] = 'raw_pandas.env'  ## Set to `cleaned_pandas.env` or `raw_pandas.env` as needed

# get_ipython().run_line_magic('run', '-i ./Data_Loading_and_Descriptive_Statistics.ipynb')
IS_TEST_DEV=True
IS_JUPYTERLAB=False


# In[2]:


if IS_TEST_DEV and IS_JUPYTERLAB:
    print("\n🎓 rawdata: [Site1 - Penrose] rawdata_site1 Dataframe & [Site2 - Takapuna] rawdata_site2 Dataframe ...")
    # rawdata


# In[3]:


if IS_TEST_DEV and IS_JUPYTERLAB:
    print("\n🎓 [Site1 - Penrose]  Summary Statistics of the {site1} rawdata_site1 Dataframe such as the mean, max/minimum values ...")
    rawdata_site1.describe()


# In[4]:


if IS_TEST_DEV and IS_JUPYTERLAB:
    print("\n🎓 [Site2 - Takapuna]  Summary Statistics of the {site2} rawdata_site2 Dataframe such as the mean, maximum and minimum values ...")
    rawdata_site2.describe()


# In[5]:


class CommonUtils:

    @staticmethod
    def calculate_metrics_adj_r2(y_true, y_pred, n_predictors):
        """
        Calculate performance metrics RMSE, MSE, MAE, MAPE, R2, and Adjusted R2 for model predictions.
        
        Args:
        y_true (array-like): Actual observed values.
        y_pred (array-like): Model's predictions.
        
        Returns:
        dict: Dictionary containing RMSE, MSE, MAE, MAPE, R2, and Adjusted R2.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ## Calculate the Mean Absolute Percentage Error (MAPE) between actual and predicted values.
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_predictors - 1)
        
        logging.debug(f"Calculated metrics - RMSE: {rmse}, MSE: {mse}, MAE: {mae}, MAPE: {mape}, R2: {r2}, Adjusted R2: {adj_r2}")
        return {'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'MAPE': mape, 'R2': r2, 'Adjusted R2': adj_r2}

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate performance metrics RMSE, MSE, MAE, MAPE, R2 for model predictions.
        
        Args:
        y_true (array-like): Actual observed values.
        y_pred (array-like): Model's predictions.
        
        Returns:
        dict: Dictionary containing RMSE, MSE, MAE, MAPE, R2, and Adjusted R2.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ## Calculate the Mean Absolute Percentage Error (MAPE) between actual and predicted values.
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
        r2 = r2_score(y_true, y_pred)
        # n = len(y_true)
        # adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_predictors - 1)

        
        logging.debug(f"Calculated metrics - RMSE: {rmse}, MSE: {mse}, MAE: {mae}, MAPE: {mape}, R2: {r2}")
        return {'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

        
    def ensure_timestamp_index(df, timestamp_col='Timestamp'):
        """
        Ensure the DataFrame uses the timestamp column as a datetime index.
        Convert a timestamp_col column to datetime and set it as index.
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Column {timestamp_col} not found in DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        # if df.index.name != timestamp_col:
        ## Retains the timestamp_col column in the DataFrame after setting it as an index.
        df.set_index(timestamp_col, inplace=True, drop=False)
    
    def reset_timestamp_index(df, timestamp_col='Timestamp'):
        """Reset DateTime Index to bring Timestamp back as a column for compatibility."""
        if df.index.name == timestamp_col:
            df.reset_index(inplace=True)
    
    def resample_data(df, timestamp_col, target_col, sample_interval='M'):
        """
        Resample/Averages time series data based on the specified interval 1M/6M/YTD/1Y/ALL. 
        Aggregate by Day/Month and Plot Daily/Monthly averages.
    
        When resampling data, setting the timestamp as the index is crucial.
        Also, after resampling, reset the index afterward for ease of further analysis.

        Parameters:
        df (DataFrame): DataFrame to resample.
        freq (str): Resampling frequency, defaults to '1D' for daily, '1h' for hourly.
        
        Returns:
            DataFrame: Resampled DataFrame with mean computed for numeric columns.
        """
        if timestamp_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"🪲 One or both specified columns: '{timestamp_col}' or '{target_col}' are not in the dataframe")
        
        ## Ensure the column for dates is in datetime format and set as index for resampling
        # if not pd.api.types.is_datetime64_any_dtype(df.index):
        #     df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        #     df.set_index(timestamp_col, inplace=True)
        ensure_timestamp_index(df, timestamp_col)    
    
        resampled_df = df[target_col].resample(sample_interval).mean()
        # resampled_df = df[target_col].resample(sample_interval).mean().reset_index()
        reset_timestamp_index(resampled_df)
        
        return resampled_df


# <div class="alert alert-block alert-info">
# 
# 🎓 The teradataml package simplifies and accelerates complex and scalable data analytics on Teradata Vantage. </br>
# 
# It bridges Python and the powerful SQL-based analytics of Teradata Vantage, enables developers to connect with Vantage via DataFrames, similar to Pandas (client-side), but performs operations on the server side. Utilising Teradata Vantage's powerful analytics capabilities, users can efficiently analyse massive datasets without having to move data from the database.
# 
# Vantage takes care of the data processing, while Python handles the input and output, combining the best of both worlds: the simplicity of Python data frames and the power of Teradata's performance and parallel processing.
# </div> 

# In[6]:


from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[7]:


class DataFrameAdapter:
    
    def __init__(self, df=None, is_teradata=False):
        """
        Initialize the adapter with an optional DataFrame and 
        a flag indicating whether it's a Teradata DataFrame.
        """
        self.is_teradata = is_teradata
        self.df = df

#     def to_pandas(self):
#         """
#         Convert the DataFrame to a pandas DataFrame if it's a Teradata DataFrame.
#         """
#         if self.is_teradata and self.df is not None:
#             return self.df.to_pandas()
#         return self.df

#     def save(self, target):
#         if self.is_teradata:
#             ## FIXME: to save Teradata DataFrame
#             save_teradata_dataframe(self.df, target)
#         else:
#             self.df.to_csv(target, index=False)

    @staticmethod
    def engineer_features(rawdata, time_col='Timestamp'):
        """
        Adds time-based features to the dataframe to enhance the analysis and exploration of seasonal patterns and dependencies.
        This includes extracting various temporal components and creating lag features for PM2.5 and PM10 variables to analyze time-based dependencies.
        
        - 'Day': The day of the month extracted from the timestamp.
        - 'DayOfWeek': The day of the week extracted from the timestamp, where Monday=0, Sunday=6.
        """

        ## Extracting date-time components/features to capture different cycles in the data.
        # timestamp_index = rawdata.index.get_level_values('Timestamp') ## 'Timestamp' <-- 'ds'
        timestamp_index = rawdata[time_col]
        rawdata['Hour']       = timestamp_index.dt.hour
        rawdata['Day']        = timestamp_index.dt.day
        rawdata['DayOfWeek']  = timestamp_index.dt.dayofweek  ## or timestamp_index.dt.weekday: Monday=0 and Sunday=6
        rawdata['Month']      = timestamp_index.dt.month
        rawdata['Quarter']    = timestamp_index.dt.quarter
        rawdata['Year']       = timestamp_index.dt.year
        # rawdata['DayOfYear']  = timestamp_index.dt.dayofyear
        
        ## Extract week of year for each timestamp
        rawdata['WeekOfYear'] = [d.isocalendar()[1] for d in timestamp_index]
        
        ## Calculating the season based on the month in New Zealand --> accurately reflects the local climate and seasonal cycles
        ## Season encoding: 1 (Spring): Sep, Oct, Nov ; 2 (Summer): Dec, Jan, Feb ; 3 (Autumn): Mar, Apr, May ; 4 (Winter): Jun, Jul, Aug
        ## TODO: Southern Hemisphere like New Zealand and Australia vs Northern Hemisphere like England
        ## Correctly mapping the month to meteorological seasons for Auckland, New Zealand
        rawdata['Season'] = rawdata['Month'].apply(
            lambda x: 1 if 9  <= x <= 11 else      ## Spring: Sep, Oct, Nov
                      2 if 12 <= x or x <= 2 else  ## Summer: Dec, Jan, Feb
                      3 if 3  <= x <= 5 else       ## Autumn: Mar, Apr, May
                      4                            ## Winter: Jun, Jul, Aug
        )
    
        # ## Weather Conditions: Create rolling averages for weather conditions to smooth out short-term fluctuations
        # rawdata['Temp_Rolling_Avg'] = rawdata['Air_Temp'].rolling(window=24, min_periods=1).mean()
        
        ## Adding lag features for PM2.5 and PM10 to capture previous time steps' influence --> to compare the correlation with the other variables.
        rawdata['PM2.5_Lag1'] = rawdata.groupby('Site')['PM2.5'].shift(1)
        rawdata['PM2.5_Lag2'] = rawdata.groupby('Site')['PM2.5'].shift(2)
        rawdata['PM10_Lag1']  = rawdata.groupby('Site')['PM10'].shift(1)
        rawdata['PM10_Lag2']  = rawdata.groupby('Site')['PM10'].shift(2)
        ## [There are NaNs in a top row's *_Lag*] Fill NaN with the next value in the column (backward fill)
        rawdata.fillna(method='bfill', inplace=True)
        ## [There are NaNs in a top row's *_Lag*] Alternatively, fill NaN with the previous value in the column (forward fill)
        rawdata.fillna(method='ffill', inplace=True)
        
        if IS_TEST_DEV:
            rawdata[[time_col, 'Day', 'DayOfWeek', 'WeekOfYear', 'Month', 'Quarter','Season', 'Year']].head()
    
        return rawdata


    @staticmethod
    def remove_highly_null_features(data, threshold=0.5):
        """
        [Display ONLY, NOT remove] Remove features with a high percentage of missing values.
        
        Args:
        data (pd.DataFrame): The dataset to be processed.
        threshold (float): The threshold for removing features (e.g., 0.5 means remove features with >50% missing values).
        
        Returns:
        pd.DataFrame: The dataset with highly null features removed.
        """
        null_fraction = data.isnull().mean()
        logging.info(f"Initial shape: {data.shape}")
        data = data.loc[:, null_fraction <= threshold]
        logging.info(f"Shape after removing highly null features: {data.shape}")
        # return data

    @staticmethod
    def remove_single_value_features(data):
        """
        [Display ONLY, NOT remove] Remove features with only one unique value.
        
        Args:
        data (pd.DataFrame): The dataset to be processed.
        
        Returns:
        pd.DataFrame: The dataset with single-value features removed.
        """
        unique_counts = data.nunique()
        logging.info(f"Initial shape: {data.shape}")
        data = data.loc[:, unique_counts > 1]
        logging.info(f"Shape after removing single-value features: {data.shape}")
        # return data

    @staticmethod
    def remove_low_information_features(data, target, threshold=0.01):
        """
        [Display ONLY, NOT remove]Remove features with low variance.
        
        Args:
        data (pd.DataFrame): The dataset to be processed.
        target (str): The target variable.
        threshold (float): The variance threshold for removing features.
        
        Returns:
        pd.DataFrame: The dataset with low information features removed.
        """
        selector = VarianceThreshold(threshold)
        features = data.drop(columns=[target])
        numeric_features = features.select_dtypes(include=[np.number])  ## Ensure only numeric features are selected
        selector.fit(numeric_features)
        low_info_features = numeric_features.columns[~selector.get_support()]
        logging.info(f"Low information features consider to drop: {low_info_features}")
        data = data.drop(columns=low_info_features)
        logging.info(f"Shape after removing low information features: {data.shape}")
        # return data
    
    @staticmethod
    def calculate_vif(data):
        """
        Calculate Variance Inflation Factor (VIF) for each feature to detect multicollinearity.
        
        Args:
        data (pd.DataFrame): The dataset containing only numerical features.
        
        Returns:
        pd.DataFrame: A DataFrame containing features and their VIF values.
        """
        vif_data = pd.DataFrame()
        vif_data['feature'] = data.columns
        vif_data['VIF'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        return vif_data

    @staticmethod
    def remove_multicollinear_features(data, threshold=5.0):
        """
        [Display ONLY, NOT remove] Remove features with high multicollinearity based on VIF.
        
        Args:
        data (pd.DataFrame): The dataset to be processed.
        threshold (float): The VIF threshold for removing features.
        
        Returns:
        pd.DataFrame: The dataset with multicollinear features removed.
        """
        vif_data = DataFrameAdapter.calculate_vif(data)
        to_drop = vif_data[vif_data['VIF'] > threshold]['feature']
        logging.info(f"Multicollinear features consider to drop: {to_drop.tolist()}")
        data = data.drop(columns=to_drop)
        logging.info(f"Shape after removing multicollinear features: {data.shape}")
        # return data
    
    @staticmethod
    def remove_highly_correlated_features(data, threshold=0.95):
        """
        [Display ONLY, NOT remove] Remove highly correlated features.
        
        Args:
        data (pd.DataFrame): The dataset to be processed.
        threshold (float): The correlation threshold for removing features.
        
        Returns:
        pd.DataFrame: The dataset with highly correlated features removed.
        """
        corr_matrix = data.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        logging.info(f"Highly correlated features consider to drop: {to_drop}")
        data = data.drop(columns=to_drop)
        logging.info(f"Shape after removing highly correlated features: {data.shape}")
        # return data

    @staticmethod
    def get_top_correlated_features(data, target, num_features=12, method='pearson'):
        """
        Calculates and returns the top N features most correlated with the target variable.
    
        Parameters:
        - data (pd.DataFrame): The dataset containing numerical features.
        - target (str): The target variable for which correlations are calculated.
        - num_features (int): The number of top features to return.
        - method (str): Method of correlation ('pearson', 'spearman', 'kendall').
    
        Returns:
        - List of tuples: Each tuple contains a feature and its correlation coefficient.
        """
        if target not in data.columns:
            raise ValueError(f"Target variable {target} not found in DataFrame.")

        ## Exclude derived or redundant features based on domain constraints
        exclude_columns = ['AQI']
        if target == 'PM10':
            exclude_columns.append('PM2.5')
            exclude_columns.append('PM2.5_Lag1')
            exclude_columns.append('PM2.5_Lag2')
        data = data.drop(columns=[col for col in exclude_columns if col in data.columns])
        logging.info(f"Excluded columns: {exclude_columns}")

        ## Select only relevant features that can be used to compute correlation; not datetime columns
        ## Ensure data does not contain infinite or NaN values
        data = data.select_dtypes(include=[np.number, np.bool_]).dropna()

        ## Remove highly null features
        # data = DataFrameAdapter.remove_highly_null_features(data)

        ## Remove single-value features
        # data = DataFrameAdapter.remove_single_value_features(data)

        ## Remove highly correlated features
        # data = DataFrameAdapter.remove_highly_correlated_features(data)
        DataFrameAdapter.remove_highly_correlated_features(data)
        # logging.info(f"Shape after removing highly correlated features: {data.shape}")

        ## Remove multicollinear features
        # data = DataFrameAdapter.remove_multicollinear_features(data)
        DataFrameAdapter.remove_multicollinear_features(data)
        # logging.info(f"Shape after removing multicollinear features: {data.shape}")
        
        ## Remove low information features
        # data = DataFrameAdapter.remove_low_information_features(data, target)
        DataFrameAdapter.remove_low_information_features(data, target)
        # logging.info(f"Shape after removing low information features: {data.shape}")
    
        ## Calculate the correlation matrix
        correlation_matrix = data.corr(method=method)

        ## Calculate absolute correlation with the target variable
        # abs_corr_with_target = abs(correlation_matrix[target])  ## NumPy
        abs_corr_with_target = correlation_matrix[target].abs()   ## Pandas

        ## Exclude the target variable and its lag variables from the series
        features_to_exclude = [target] + [f"{target}_Lag1", f"{target}_Lag2"]
        abs_corr_with_target = abs_corr_with_target.drop(features_to_exclude, errors='ignore')
    
        # Get the top N highly correlated features
        # top_features = abs_corr_with_target.nlargest(num_features).index.tolist()
        top_features = abs_corr_with_target.nlargest(num_features)
    
        ## Convert top features from Series to list of tuples
        return [(feature, correlation) for feature, correlation in top_features.items()]

        
 
    @staticmethod
    def compute_summary_statistics(data, target, top_features):
        """
        Computes and prints summary statistics for selected features in a given dataset.
        
        Args:
        data (pd.DataFrame): The cleaned data for a specific site.
        top_features (list): List of features for which to compute statistics, determined from prior analysis.
        
        Returns:
        pd.DataFrame: Summary statistics of the selected features.
        """
        ## Ensure all top features are in the data columns
        valid_features = [feature for feature in top_features if feature in data.columns]
        ## Note: Create a new DataFrame with the scaled features, using the original numeric column names ONLY
        # valid_features = ['Timestamp'] + valid_features
        
        selected_variables = [target] + valid_features
        ## Get summary statistics
        summary_stats = data[selected_variables].describe()
        
        return valid_features, summary_stats
    
    @staticmethod
    def scale_features(self, features):
        """
        Scaling Features: If scaling is required for certain machine learning models, this method prepares the scaled dataframe.
    
        Scales features to be used in machine learning and deep learning models.
        """
        ## Create a StandardScaler instance
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        
        ## Fit the scaler to the numeric features and transform
        scaled_features = scaler.fit_transform(features)
        
        ## Create a new DataFrame with the scaled features and original feature names
        scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)
        
        ## Set the 'Timestamp' column as the index of the DataFrame
        # scaled_features_df.set_index('Timestamp', inplace=True)

        return scaled_features_df


    def train_test_split(self, target, shuffle=False):
        """
        Training and Testing Split: Methods for splitting the data into training and testing sets are implemented, with options of using random splits or a cutoff date.
        Splits the data into training and testing sets.
        """
        X = self.df_scaled.drop(columns=[target])
        y = self.df_scaled[target]
        return train_test_split(X, y, test_size=0.2, shuffle=shuffle)

# ## Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(scaled_features_df, target, test_size=0.2)


# In[8]:


# # !pip install tsfresh
# # !pip install featuretools

# from tsfresh import extract_features, select_features
# from tsfresh.utilities.dataframe_functions import impute
# from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters
# from tsfresh.feature_selection.relevance import calculate_relevance_table

# import featuretools as ft
# from featuretools import selection
# from featuretools.selection import remove_highly_null_features, remove_single_value_features, remove_highly_correlated_features, remove_low_information_features


# In[9]:


# class FeatureEngineering:

    # @staticmethod
    # def extract_tsfresh_features(data, target_column):
    #     """
    #     Extracts features using Tsfresh from the provided data.
    
    #     Args:
    #     data (pd.DataFrame): The time series data.
    #     target_column (str): The target variable for which features need to be extracted.
    
    #     Returns:
    #     pd.DataFrame: DataFrame with extracted features.
    #     """
    #     logging.info(f"Starting feature extraction for target: {target_column}")
    
    #     ## Reset index for tsfresh compatibility
    #     data.reset_index(inplace=True)
        
    #     ## Define extraction settings
    #     # extraction_settings = ComprehensiveFCParameters()
    #     extraction_settings = EfficientFCParameters()
    
    #     ## Extract features
    #     extracted_features = extract_features(data, column_id='id', column_sort='Timestamp', 
    #                                           default_fc_parameters=extraction_settings,
    #                                           disable_progressbar=False, n_jobs=4)

    #     ## Log extraction settings
    #     logging.debug(f"Extraction settings: {extraction_settings}")
        
    #     ## Impute missing values
    #     impute(extracted_features)
    #     logging.info(f"Features extracted: {extracted_features.shape[1]} features.")
    
    #     ## Select relevant features
    #     relevant_features = select_features(extracted_features, data[target_column])
    #     logging.info(f"Relevant features selected: {relevant_features.shape[1]} features.")
        
    
    #     ## Merge with original data for further processing
    #     data = data.merge(relevant_features, left_index=True, right_index=True, how='inner')
    #     logging.info(f"Data merged with selected features: {data.shape}.")
    
    #     logging.info(f"Feature extraction and selection completed for target: {target_column}")

    #     return data

    
    # @staticmethod
    # def extract_featuretools_features(data, target_column, entity_id):
    #     """
    #     Extracts features using Featuretools from the provided data.
        
    #     Args:
    #     data (pd.DataFrame): The time series data.
    #     target_column (str): The target variable for which features need to be extracted.
    #     entity_id (str): Unique identifier for the entity set.

    #     Returns:
    #     pd.DataFrame: DataFrame with extracted features.
    #     """
    #     logging.info(f"Starting feature extraction for target: {target_column}")

    #     ## Ensure the index is properly set and create a unique identifier
    #     data['id'] = data.index

    #     ## Create an entity set
    #     es = ft.EntitySet(id=entity_id)
        
    #     ## Add the data as an entity
    #     es.add_dataframe(dataframe_name=entity_id, dataframe=data, 
    #                      index='id', time_index='Timestamp')
        
    #     ## Run deep feature synthesis with transformation primitives
    #     feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name=entity_id, 
    #                                           agg_primitives=['mean', 'sum', 'std', 'max', 'min'],
    #                                           # trans_primitives=['day', 'weekday', 'month', 'year'],
    #                                           trans_primitives=['day', 'weekday', 'month', 'year', 'cum_sum', 'time_since_previous'],
    #                                           max_depth=2)
        
    #     logging.info(f"Feature matrix shape: {feature_matrix.shape}")

    #     ## Remove any features that are not relevant to the target
    #     # y = data[target_column]
    #     # X_filtered = selection.remove_low_information_features(X_filtered, y)
    #     X_filtered = selection.remove_highly_null_features(feature_matrix)
    #     X_filtered = selection.remove_single_value_features(X_filtered)
    #     X_filtered = selection.remove_highly_correlated_features(X_filtered, pct_corr_threshold=0.95)
    #     X_filtered = selection.remove_low_information_features(X_filtered)
        
    #     logging.info(f"Filtered feature matrix shape: {X_filtered.shape}")

    #     ## Merge with original data for further processing
    #     data = data.merge(X_filtered, left_index=True, right_index=True, how='inner')
    #     logging.info(f"Data merged with selected features: {data.shape}.")
    
    #     logging.info(f"Feature extraction and selection completed for target: {target_column}")

    #     return data


# In[10]:


## Usage:
# dataFrameAdapter = DataFrameAdapter(is_teradata=IS_TERADATA_VANTAGE)
# adapter.read_csv("data.csv")
# pandas_df = dataFrameAdapter.to_pandas()

if IS_JUPYTERLAB:
    ## Adding time-based features to the dataframeto enhance the analysis and exploration of seasonal patterns and dependencies.
    cleaned_data = DataFrameAdapter.engineer_features(rawdata.copy())
    cleaned_data


# In[11]:


if IS_JUPYTERLAB:
    ## 'Timestamp' as the only Ordinal Attribute/Column given its nature order in time-series data
    cleaned_ordinal_columns = ['Timestamp']
    
    print("\n🎓 Describing the types of each attribute as cleaned_numerical_columns (Continuous), cleaned_ordinal_columns (Ordinal), or cleaned_nominal_columns (Nominal) ...")
    cleaned_numerical_columns, cleaned_nominal_columns = DescriptiveStatistics.describe_data(cleaned_data, cleaned_ordinal_columns)


# In[12]:


if IS_JUPYTERLAB:
    cleaned_data_site1 = cleaned_data[cleaned_data['Site'] == 'Penrose']
    cleaned_data_site2 = cleaned_data[cleaned_data['Site'] == 'Takapuna']
    
    # if IS_TEST_DEV:
    print("\n🎓 [Site1 - Penrose]  Summary Statistics of the {site1} cleaned_data_site1 Dataframe such as the mean, max/minimum values ...")
    cleaned_data_site1.describe()   


# In[13]:


if IS_JUPYTERLAB:
    print("\n🎓 [Site2 - Takapuna]  Summary Statistics of the {site2} cleaned_data_site2 Dataframe such as the mean, max/minimum values ...")
    cleaned_data_site2.describe()


# In[14]:


from abc import ABC, abstractmethod
class DataLoader(ABC):
    """
    Abstract base class define the interface for data loaders.
    """

    @abstractmethod
    def load_data(self, source):
        """
        Loads data from the specified source.
        """
        pass


class PandasDataLoader(DataLoader):
    """
    Concrete implementation for loading data using pandas.
    """
    # @staticmethod
    # def load_data(self, file_path):
    #     """
    #     Loads data from a file into a pandas DataFrame.
    #     """
    #     return pd.read_csv(file_path)

    @staticmethod
    def load_data(pandas_df):
        """
        Loads data into a pandas DataFrame. 
        The 'Timestamp' --> 'ds' column is properly converted to a DateTime index for time series analysis.
        """
        ## Ensure the 'Timestamp' is the index and convert it to datetime if not already
        if pandas_df.index.name != 'Timestamp' or not pd.api.types.is_datetime64_any_dtype(pandas_df.index):
            pandas_df.rename(columns={'Timestamp': 'ds'}, inplace=True) ## Compatible with Prophet
            pandas_df = pandas_df.set_index('ds')
            pandas_df.index = pd.to_datetime(pandas_df.index)
        else:
            raise ValueError("DataFrame must contain a 'Timestamp' column.")
        
        return pandas_df


class TeradataDataLoader(DataLoader):
    """
    Concrete implementation for loading data using teradataml.
    """
    # @staticmethod
    # def load_data(self, table_name):
    #     """
    #     Loads data from a Teradata table into a teradataml DataFrame.
    #     """
    #     return tdml.DataFrame(table_name)

    @staticmethod
    def load_data(self, pandas_df):
        """
        Loads data from a pandas DataFrame into a teradataml DataFrame.
        """
        teradata_df = tdml.DataFrame.from_pandas(pandas_df)
        ## In reality, we might be reading from a Teradata table.
        # teradata_df = fastload(df = pandas_df, table_name = 'my_table')
        
        return teradata_df


# In[15]:


## ONLY available in 4_Time_Series_Forecasting.ipynb

# if IS_JUPYTERLAB:
#     target_variables = {'PM2.5': 'Particulate Matter <2.5 µm', 'PM10': 'Particulate Matter <10 µm'}
#     data_series_df1 = PandasDataLoader.load_data(rawdata_site1)
#     data_series_df2 = PandasDataLoader.load_data(rawdata_site2)
#    
#     data_series_df1


# In[16]:


from prettytable import PrettyTable

def display_cleaned_site_comparison(is_include_rawdata=True):
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
    if is_include_rawdata:
        data_dictionary = [
            ("rawdata", "Complete dataset containing all observations across all sites.", "[x]", "[x]", "[x]"),
            ("numerical_columns_site1", "Numerical columns specific to Site 1.", "[ ]", "[x]", "[ ]"),
            ("nominal_columns_site1", "Nominal columns specific to Site 1.", "[ ]", "[x]", "[ ]"),
            ("numerical_columns_site2", "Numerical columns specific to Site 2.", "[ ]", "[ ]", "[x]"),
            ("nominal_columns_site2", "Nominal columns specific to Site 2.", "[ ]", "[ ]", "[x]"),
            ("rawdata_site1", "Subset of raw data for Site 1.", "[ ]", "[x]", "[ ]"),
            ("rawdata_site2", "Subset of raw data for Site 2.", "[ ]", "[ ]", "[x]"),
            ("---------------------------", "---------------------------------------------------------------------", "---------", "-------", "--------"),  ## Blank line for separation
            ("cleaned_data", "Cleaned dataset with preprocessing applied.", "[x]", "[x]", "[x]"),
            ("cleaned_ordinal_columns", "Ordinal columns in the cleaned dataset.", "[x]", "[x]", "[x]"),
            ("cleaned_numerical_columns", "Numerical columns in the cleaned dataset.", "[x]", "[x]", "[x]"),
            ("cleaned_nominal_columns", "Nominal columns in the cleaned dataset.", "[x]", "[x]", "[x]"),
            ("cleaned_data_site1", "Cleaned data for Site 1.", "[ ]", "[x]", "[ ]"),
            ("cleaned_data_site2", "Cleaned data for Site 2.", "[ ]", "[ ]", "[x]"),
            # ("---------------------------", "---------------------------------------------------------------------", "---------", "-------", "--------"),  ## Blank line for separation
            # ("imputed_cleaned_data_site1", "Imputed Cleaned data for Site 1.", "[ ]", "[x]", "[ ]"),
            # ("imputed_cleaned_data_site2", "Imputed Cleaned data for Site 2.", "[ ]", "[ ]", "[x]"),
            # ("---------------------------", "---------------------------------------------------------------------", "---------", "-------", "--------"),  ## Blank line for separation
            # ("winsorized_imputed_cleaned_data_site1", "Winsorized Imputed Cleaned data for Site 1.", "[ ]", "[x]", "[ ]"),
            # ("winsorized_imputed_cleaned_data_site2", "Winsorized Imputed Cleaned data for Site 2.", "[ ]", "[ ]", "[x]"),
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
    else:
        data_dictionary = [
            ("rawdata", "Complete dataset containing all observations across all sites.", "[x]", "[x]", "[x]"),
            ("include_columns_site1", "Numerical columns specific to Site 1.", "[ ]", "[x]", "[ ]"),
            ("include_columns_site2", "Numerical columns specific to Site 2.", "[ ]", "[ ]", "[x]"),
            ("---------------------------", "---------------------------------------------------------------------", "---------", "-------", "--------"),  ## Blank line for separation
            ("cleaned_data", "Cleaned dataset with preprocessing applied.", "[x]", "[x]", "[x]"),
            ("cleaned_ordinal_columns", "Ordinal columns in the cleaned dataset.", "[x]", "[x]", "[x]"),
            ("cleaned_numerical_columns", "Numerical columns in the cleaned dataset.", "[x]", "[x]", "[x]"),
            ("cleaned_nominal_columns", "Nominal columns in the cleaned dataset.", "[x]", "[x]", "[x]"),
            ("cleaned_data_site1", "Cleaned data for Site 1.", "[ ]", "[x]", "[ ]"),
            ("cleaned_data_site2", "Cleaned data for Site 2.", "[ ]", "[ ]", "[x]"),
        ]

    ## Format for better readability
    for entry in data_dictionary:
        ## Adding rows to the table
        table.add_row(entry)

    ## Print the table in an organized format
    print(table)

if IS_JUPYTERLAB:
    ## Call the function to display the table
    print('\n🎓 [DataFrameAdapter.ipynb] Listing variables with description...')
    display_cleaned_site_comparison(is_include_rawdata=True)


# <footer style="padding-bottom:35px; background:#f9f9f9; border-bottom:3px solid #00b2b1">
#     <div style="float:left;margin-top:14px;color:#E37C4D">Predicting Air Particulate Matter at Scale ⛅️</div>
#     <div style="float:right;">
#         <div style="float:left; margin-top:14px">
#             Auckland University of Technology (AUT) 🎓
#         </div>
#     </div>
# </footer>
