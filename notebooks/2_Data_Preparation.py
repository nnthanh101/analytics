#!/usr/bin/env python
# coding: utf-8

# <header style="padding:1px;background:#f9f9f9;border-top:3px solid #00b2b1"><img id="Teradata-logo" src="https://www.teradata.com/Teradata/Images/Rebrand/Teradata_logo-two_color.png" alt="Teradata" width="220" align="right" />
# 
# <b style='font-size:28px;font-family:Arial;color:#E37C4D'>🎓 Predicting Air Particulate Matter at Scale ⛅️</b><br>
# <b style='font-size:28px;font-family:Arial;color:#E37C4D'>🛠️ 2. Data Preparation & Exploratory Data Analysis</b>
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
# 🎓 This notebook shows various DataFrame methods that can be used to analyse and cleanse a dataset. It should be ready for reuse in the next steps (Time Series, Machine Learning, Deep Learning) in CRISP-DM for Data Science
#     
# </div>

# * **Workflow steps:**
# 
#   1. Import the required teradataml modules.
#   2. Connect to a Vantage system.
#   3. Data Loading and visualize the data using Plot(). 
#   4. Data Analysis & preparation e.g. use of various dataframe functions to get details about the data like shape, null values etc., use Variable transformation to fill NULL values. 
#   5. Cleanup.

# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
#     
# # 🎯 Libraries and Reusable Functions 

# <div class="alert alert-block alert-info">
# 🎓 To execute all of the cells in `Data_Loading_and_Descriptive_Statistics.ipynb`.
# </div>

# In[9]:


import logging

## TODO: .env --> determines the environment for output format programmatically
## Check for the JupyterLab environment, which might affect how visualizations are rendered or interacted with.
IS_JUPYTERLAB               = True  ## True if JupyterLab; False if Python .py
IS_TERADATA_VANTAGE         = False ## True if Data in Teradata Vantage; False if Laptop/Virtual-Machine
IS_DATA_IN_TERADATA_VANTAGE = False ## True if Data in Teradata Vantage; False if Data from *.csv/*xls files
IS_DEBUG                    = True  ## Plot and display additional information or not

# if not IS_DEBUG:
## Set logging level to WARNING to suppress info messages --> turn-off Prophet logs
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
# logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)

get_ipython().run_line_magic('run', '-i ./Data_Loading_and_Descriptive_Statistics.ipynb')


# In[10]:


print("\n🎓 [Site1 - Penrose]  Summary Statistics of the {site1} rawdata_site1 Dataframe such as the mean, max/minimum values ...")
rawdata_site1.describe()


# In[11]:


print("\n🎓 [Site2 - Takapuna]  Summary Statistics of the {site2} rawdata_site2 Dataframe such as the mean, maximum and minimum values ...")
rawdata_site2.describe()


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## Python Reusable Functions

# <div class="alert alert-block alert-info">
# 🎓 This section will examine the key statistics and missing data percentages from the reports to gain a better understanding of the data characteristics and time-series features.
# </div>

# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # ⚙️ Data Preprocessing

# <div class="alert alert-block alert-info">
# 🎓 This section handles missing data by using function wrappers (also known as decorators) to modify and extend an existing imputation function. Additionally, filter and display potential outliers for further investigation and analysis.
# </div>

# In[12]:


## [DEBUG]
# rawdata.columns

## [OPTION] Remove not usefull columns
# cleaned_data = cleaned_data.drop(['CO', 'O3', 'Solar_Rad'], axis=1)


# In[13]:


# def ensure_datetime_format(df, column_name='Timestamp'):
#     """
#     Ensures the specified column is in datetime format.
#     Parameters:
#     - df (pd.DataFrame): The dataframe to process.
#     - column_name (str): The name of the column to format as datetime.
#     Returns:
#     - pd.DataFrame: DataFrame with the column in datetime format.
#     """
#     try:
#         df[column_name] = pd.to_datetime(df[column_name])
#     except Exception as e:
#         raise ValueError(f"Error converting {column_name} to datetime: {e}")
#     return df

def clean_timeseries_dataset(df, time_col='Timestamp'):
    """
    Cleans the dataset by ensuring datetime format for 'Timestamp', 
    removing rows where all values are NaN (except time_col),
    and removing duplicate rows based on 'Timestamp'.
    Parameters:
    - df (pd.DataFrame): Input dataset.
    - time_col (str): The column name for timestamp data.
    Returns:
    - pd.DataFrame: Cleaned dataset.
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame.")
    
    ## Convert 'Timestamp' column to datetime format
    # df = ensure_datetime_format(df, column_name=time_col)

    ## Remove rows where all values are NaN, excluding 'Timestamp'
    cols_except_time = [col for col in df.columns if col != time_col]
    if IS_DEBUG:
        rows_all_nans = df[df[cols_except_time].isna().all(axis=1)]
        if not rows_all_nans.empty:
            print("Rows where all values are NaN, excluding 'Timestamp':\n", rows_all_nans)
    ## Remove rows where all values are NaN, excluding 'Timestamp'
    df = df.dropna(how='all', subset=cols_except_time)

    if IS_DEBUG:
        ## Show duplicate rows based on 'Timestamp'
        duplicates = df[df.duplicated(subset=[time_col], keep=False)]
        if not duplicates.empty:
            print("Duplicate rows based on 'Timestamp':\n", duplicates)
    
    ## Remove duplicate rows based on 'Timestamp'
    df = df.drop_duplicates(subset=[time_col], keep='first')

    ## Next Steps:
    # - Handling outliers in numeric columns
    # - Filling missing values with appropriate imputation techniques
    # - Normalizing or scaling numeric columns
    # - Encoding categorical variables if necessary

    ## Optional: sort by 'Timestamp' for chronological order
    # cleaned_df.sort_values(by=time_col, inplace=True)

    return df


# In[68]:


cleaned_data_site1 = clean_timeseries_dataset(rawdata_site1)
print(f"☑️          rawdata_site1 dataset shape: {rawdata_site1.shape}")
print(f"✅ Cleaned rawdata_site1 dataset shape: {cleaned_data_site1.shape}")

cleaned_data_site2 = clean_timeseries_dataset(rawdata_site2)
print(f"☑️          rawdata_site2 dataset shape: {cleaned_data_site2.shape}")
print(f"✅ Cleaned rawdata_site2 dataset shape: {cleaned_data_site2.shape}")


# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Missing Data Imputation in Time-Series

# <header style="padding:1px;background:#00b2b1">
# 
# 🎓 Using **function wrappers** (also known as **decorators**) to modify and extend an existing imputation function. Decorators can indeed be a effective tool to modify and extend the behavior of functions, including for tasks like data imputation. Each decorator does the following actions: (1) accepts a function as an input argument (which, when called/invoked, returns a DataFrame); and (2) returns a new function that, when called/invoked, also returns a DataFrame but with imputed data depending on the specified given imputation approach.
# 
# 0. ~~**Option 0**: Fill NaN with Outlier or Zero~~
# 
# In this specific example filling the missing value with an outlier value such as np.inf or 0 seems to be very naive. However, using values like -999, is sometimes a good idea.
# 
# 1. **Option 1**: Fill NaN with Mean or Mode Value
# 
# Filling NaNs with the mean value is also not sufficient and naive, and doesn't seems to be a good option.
# 
# 2. **Option 2**: Fill NaN with Last Value with .ffill()
# 
# Filling NaNs with the last value could be bit better.
# 
# 3. **Option 3**: Fill NaN with Linearly Interpolated Value with .interpolate()
# 
# Filling NaNs with the interpolated values is the best option in this small examlple but it requires knowledge of the neighouring value 
# 
# 4. **Option 4**: Fill NaN with Time-Series Moving Average (TODO)
# 
# Filling NaNs with ... TBD
# 
# 5. **Option 5**: Fill NaN with Time-Series Model (Facebook/Meta Prophet)
# 
# 6. **Option 6**: Fill NaN with Machine Learning Model (XGBoost)
# 
# 7. ~~**Option N**: Filling NaNs with something else (TBD)~~
# 
# </header>

# In[70]:


# !pip install --upgrade prophet xgboost --user
import functools
import holidays
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor


# In[71]:


class DataImputer:
    """
    Class to perform various imputation methods on a DataFrame.
    Note: Imputation methods should directly accept a DataFrame and return the imputed DataFrame
    """
    def __init__(self):
        # if not isinstance(data, pd.DataFrame):
        #     raise ValueError("Data must be a pandas DataFrame")
        pass

    @staticmethod
    def mean_mode_imputation(data, columns=None):
        """
        Impute with Mean and Mode:
        Replaces missing numerical values with the MEAN and missing categorical values with the MODE.

        Args:
            data (DataFrame): DataFrame containing the data.
            columns (list, optional): Columns to impute. If None, applies to all columns.

        Returns:
            DataFrame: DataFrame with imputed values.
        """
        if columns is None:
            # Exclude 'Timestamp' column or any non-numeric columns
            columns = [col for col in data.columns if data[col].dtype in ['float64', 'int64'] and col != 'Timestamp']
        
        # print("Applying Mean/Mode Imputation...")
        for col in data.columns:
            if col == 'Timestamp':  # Skip the 'Timestamp' column
                continue

            if data[col].dtype in ['float64', 'int64']:
                imputer = SimpleImputer(strategy='mean')
            else:  ## Assuming categorical data: return_value[col].dtype.name == 'category'
                imputer = SimpleImputer(strategy='most_frequent')
            ## The fit_transform function expects a 2D array, hence the double square brackets
            data[col] = imputer.fit_transform(data[[col]])
            
        return data

    @staticmethod
    def forward_backward_imputation(data):
        """
        Impute with Forward and Backward propagation
        Using ffill and bfill as a naive method, to complete the data.

        Parameters:
            data (pd.DataFrame): DataFrame to impute.
            inplace (bool): If False, returns a new DataFrame, otherwise modifies the original DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """
        # print("Applying Forward and Backward Imputation across the entire DataFrame ...")
        ## Forward fill (ffill) missing values
        data.fillna(method='ffill', inplace=True)
        ## Backward fill (bfill) remaining missing values
        data.fillna(method='bfill', inplace=True)
        return data

    @staticmethod
    def interpolation_imputation(data, method='linear'):
        """
        Impute with Interpolation
        A linear interpolation method works by assuming a linear relationship between the observed points and drawing a straight line accordingly.
        Polynomial curves or splines.
        Interpolation is an effective approach to impute missing values in time series. 
        Polynomial interpolation fits a polynomial function to the observed data points and estimates the missing values based on this function.
        It works best if the time series is reasonably smooth. In case there are sudden changes or outliers, a simpler approach such as forward filling might be a better option.

        Parameters:
            data (pd.DataFrame): DataFrame to impute.
            method (str): Interpolation technique, default is 'linear'.
            order (int, optional): The order of the polynomial for polynomial interpolation.
            inplace (bool): If False, returns a new DataFrame, otherwise modifies the original DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """
        # print("Applying Interpolation Imputation...")
        if method == 'polynomial' and order is None:
            raise ValueError("Order must be specified for polynomial interpolation")
        data.interpolate(method=method, inplace=True, limit_direction='both')
        return data

    @staticmethod
    def polynomial_imputation(data, method='polynomial', order=2):
        """
        Impute with Interpolation
        A linear interpolation method works by assuming a linear relationship between the observed points and drawing a straight line accordingly.
        Polynomial curves or splines.
        Interpolation is an effective approach to impute missing values in time series. 
        Polynomial interpolation fits a polynomial function to the observed data points and estimates the missing values based on this function.
        It works best if the time series is reasonably smooth. In case there are sudden changes or outliers, a simpler approach such as forward filling might be a better option.
        """
        # print("Applying Polynomial Imputation...")
        if method == 'polynomial' and order is None:
            raise ValueError("Order must be specified for polynomial interpolation")
        # data.interpolate(method=method, order=order if order else None, inplace=True, limit_direction='both')
        ## Apply polynomial interpolation across columns
        for column in data.select_dtypes(include=['float', 'int']).columns:
            ## Only interpolate if there are at least two non-NA values
            if data[column].count() > 1:
                data[column].interpolate(method='polynomial', order=order, inplace=True, limit_direction='both')
        # return_value = return_value.interpolate(method='spline')
        return data
    

    @staticmethod
    def moving_average_imputation(data, columns=None, window='168H', min_periods=1, verbose=True):
    # def moving_average_imputation(data, columns=None, window='168H', min_periods=1, verbose=False, inplace=True):
        """
        Impute missing values using a moving average. 
        Only NaNs are filled with the rolling mean, preserving the integrity of original non-NaN values. 
        A moving average is better at adapting to changes by considering a few nearby data points to compute the mean.
        Yet, it can still lead to biased results if the data is not missing at random.
        
        Examples:
            .rolling(window='52W' ## A year has 52 weeks (52 weeks * 7 days per week) approximately.
            .rolling(window='168H', min_periods=1).mean()  ## 168 hours = 7 days

        Parameters:
            data (DataFrame): The pandas DataFrame containing the data.
            columns (list of str, optional): Specific columns to impute. If None, all numeric columns are used.
            window (str): The size of the moving window (default is '168H' for 168 hours or 7 days).
            min_periods (int): Minimum number of observations in window required to have a value (default is 1).
            verbose (bool): If True, print additional information about the imputation process.
            # inplace (bool): If True, modify the DataFrame in place. Otherwise, return a modified copy.

        Returns:
            DataFrame: The DataFrame with imputed values if inplace is False. Otherwise, None.

        Notes:
            - A rolling window imputation can adapt to changes by considering nearby data points. However, it may
              introduce biases if the data is not missing at random.
            - This method currently supports only numeric data for rolling calculations.
        """
        ## Ensure the 'Timestamp' column is in datetime format and set as index only if it's not already the index
        if 'Timestamp' in data.columns:
            data['Timestamp'] = pd.to_datetime(data['Timestamp'], dayfirst=True, errors='coerce')
            data.set_index('Timestamp', inplace=True)

        if not pd.api.types.is_datetime64_any_dtype(data.index):
            raise ValueError("Data index must be a datetime type for time-based rolling windows.")

        ## Select columns if not provided
        if columns is None:
            ## Select only numeric columns for rolling operation
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
        if verbose:
            print("Applying Moving Average Imputation on columns:", columns)

        ## Apply rolling mean to specified numeric columns
        for column in columns:
            ## Check if the column is of a numeric type: boolean (b), integer (i), floating (f), or complex number (c)
            if data[column].dtype.kind in 'bifc':  
                if verbose:
                    print(f"Processing column: {column}")
                ## Calculate the rolling mean and update NaNs only
                rolling_mean = data[column].rolling(window=window, min_periods=min_periods).mean()
                data[column].fillna(rolling_mean, inplace=True)

        ## Iterative pass to fill remaining NaNs: 
        ## Perform multiple passes over the data. In each pass, it tries to fill the NaNs with the rolling mean. 
        max_iterations = 10
        for iteration in range(max_iterations):
            change_made = False
            for column in columns:
                if data[column].isna().sum() > 0:  ## Check if there are still NaNs to fill
                    if verbose:
                        print(f"Iterative imputation for column: {column}, iteration {iteration+1}")
                    before_fill = data[column].isna().sum()
                    data[column].fillna(
                        data[column].rolling(window=window, min_periods=min_periods).mean(),
                        inplace=True
                    )
                    after_fill = data[column].isna().sum()
                    if after_fill < before_fill:
                        change_made = True
            
            if not change_made:  ## No more NaNs can be filled
                if verbose:
                    print(f"No more changes made in iteration {iteration+1}. Ending iterative imputation.")
                break
        
        ## Reset index to turn 'Timestamp' back into a column
        data.reset_index(inplace=True)
        
        return data


    @staticmethod
    def prophet_imputation(data, date_column, target_columns, holidays=None):
        """
        Decorator for imputing missing values using Facebook/Meta Prophet time series forecasting model.
        
        Parameters:
        - dates_column: Name of the column containing the dates.
        - columns_to_impute: Columns to impute. If 'all', all columns except the dates column are imputed.
        
        Returns:
        - A wrapper function for the imputation.
        """
        # print("Applying Time-Series Prophet Imputation...")    
        if date_column not in data.columns:
            raise ValueError(f"{date_column} is not a column in the DataFrame")

        for column in target_columns:
            ## Prepare DataFrame for Prophet
            df_prophet = pd.DataFrame({
                'ds': data[date_column],
                'y': data[column],
                # 'y': data[column].interpolate()  ## Improved handling of missing values ?
            }).dropna()
            
            ## Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality='auto',
                holidays=holidays
            )
            model.fit(df_prophet)
            ## TODO: Make future dataframe and predict
            # future = model.make_future_dataframe(periods=0, freq='H')
            # forecast = model.predict(future)
            forecast = model.predict(df_prophet[['ds']])
            ## Fill missing values in original data
            forecast.set_index('ds', inplace=True)
            data.set_index(date_column, inplace=True)
            # data[column].update(forecast)
            ## 'yhat' from Prophet forecast can be used as imputed values
            data[column].fillna(forecast['yhat'], inplace=True)
            data.reset_index(inplace=True)
        return data

    @staticmethod
    def xgboost_imputation(data, target_columns):
        """
        Impute missing values in a DataFrame using XGBoost with Scikit-learn's IterativeImputer.
        
        Parameters:
        - data: DataFrame to be imputed.
        - target_columns: Columns to impute, could be 'all' or a list of column names.
        
        Returns:
        - The DataFrame with imputed values.
        """
        # print("Applying Machine-Learning XGBoost Interpolation Imputation...")
        ## TODO: Set default estimator if none provided -> Directly use XGBRegressor as the estimator
        # chosen_estimator = estimator if estimator is not None else XGBRegressor(n_estimators=100, random_state=0)
        chosen_estimator = XGBRegressor(n_estimators=100, random_state=0)
        ## Perform imputation on numerical data
        imputer = IterativeImputer(estimator=chosen_estimator, max_iter=10, random_state=0)
        ## TODO: Apply imputer only on specified target columns
        # data[target_columns] = imputer.fit_transform(data[target_columns])
        # for column in target_columns:
        #     if column not in data.columns:
        #         raise ValueError(f"{column} is not a column in the DataFrame")
        #     data[[column]] = imputer.fit_transform(data[[column]])
        # Perform imputation on specified target columns or all numerical columns if 'all'
        if target_columns == 'all':
            numerical_data = data.select_dtypes(include=[np.number])
            imputed_numerical_data = pd.DataFrame(IterativeImputer(estimator=chosen_estimator, max_iter=10, random_state=0).fit_transform(numerical_data), columns=numerical_data.columns, index=data.index)
            data.update(imputed_numerical_data)
        else:
            for column in target_columns:
                if column in data.columns:
                    column_data = data[[column]].select_dtypes(include=[np.number])
                    imputed_column_data = IterativeImputer(estimator=chosen_estimator, max_iter=10, random_state=0).fit_transform(column_data)
                    data[column] = imputed_column_data
        return data

# ### WIP

#     @staticmethod
#     def prophet_imputation(data, date_column, target_columns, holidays=None):
#         data[date_column] = pd.to_datetime(data[date_column])
#         data = data.set_index(date_column)

#         for column in target_columns:
#             # Add additional regressors for pollutant and meteorological data
#             additional_regressors = [col for col in data.columns if col not in [column, date_column]]

#             model = Prophet(
#                 daily_seasonality=True,
#                 weekly_seasonality=True,
#                 yearly_seasonality='auto',
#                 holidays=holidays
#             )
#             # Add meteorological and pollutant data as additional regressors
#             for reg in additional_regressors:
#                 model.add_regressor(reg)

#             df_prophet = data.reset_index()[[date_column, column] + additional_regressors].dropna()
#             model.fit(df_prophet)
#             forecast = model.predict(df_prophet.drop(columns=[column]))

#             # Impute only missing values
#             imputed_values = forecast.set_index('ds')['yhat']
#             data[column] = data[column].combine_first(imputed_values)

#         data.reset_index(inplace=True)
#         return data

#     @staticmethod
#     def xgboost_imputation(data, target_columns, max_iter=10):
#         if target_columns == 'all':
#             target_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
#         for column in target_columns:
#             if column in data.columns:
#                 # Prepare features including lagged and time-based features
#                 data = DataImputer.prepare_features(data, column)

#                 imputer = IterativeImputer(
#                     estimator=XGBRegressor(n_estimators=100, random_state=0),
#                     max_iter=max_iter,
#                     random_state=0
#                 )
#                 # Impute only NaN values
#                 data[column] = data[column].combine_first(pd.Series(imputer.fit_transform(data.drop(columns=[column])).ravel(), index=data.index))

#         return data

#     @staticmethod
#     def prepare_features(data, target_column):
#         """
#         Prepares lagged and time-based features for a given column.
#         """
#         data['hour'] = data.index.hour
#         data['dayofweek'] = data.index.dayofweek
#         # Add more time-based features as needed
#         # Add lagged features
#         data[f'{target_column}_lag1'] = data[target_column].shift(1)
#         data[f'{target_column}_lag2'] = data[target_column].shift(2)
#         # Add more lagged features as needed

#         # Ensure no NaN values in the new feature columns
#         data.fillna(method='bfill', inplace=True)
#         data.fillna(method='ffill', inplace=True)

#         return data


# In[72]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

def before_and_after_values_visualization(data_site, imputed_data, method_description, numerical_columns, site, feature_column='PM10'):
    """
    This function plot the comparison between the two sites before and after imputation, adjusting parameters based on the visualization type.
    Applies the specified imputation function to the data, then visualizes the results using the specified visualization function.
    
    Parameters:
    - data: pd.DataFrame - The dataset to impute and visualize.
    - imputed_data: pd.DataFrame - The imputed dataset.
    - site: str - The name of the site/location for comparison.
    - feature_column: str - The feature column on which to focus the visualization.
    """
    if IS_DEBUG:
        print(f"before_and_after_values_visualization {data_site.shape}:", imputed_data.shape, site, visualization_type, feature_column)
    if not isinstance(data_site, pd.DataFrame):  ## Ensure data is a DataFrame before proceeding
        raise ValueError("Data must be a pandas DataFrame.")

    ## Plotting using Plotly: Setting up the figure to hold two subplots side by side
    fig = make_subplots(rows=1, cols=1, shared_yaxes=True, subplot_titles=(f'Data Plot of {feature_column} at {site}'))

    ## Original and Modified Data for Site1: darkorange vs lightblue | dodgerblue; dash='dot' vs 'solid'
    fig.add_trace(go.Scatter(x=imputed_data['Timestamp'], y=imputed_data[feature_column], mode='markers', name='Modified Site', 
                             line=dict(color='darkorange', width=0.3, dash='dot')), row=1, col=1)
    # fig.add_trace(go.Scatter(x=data_site['Timestamp'], y=data_site[feature_column], mode='lines+markers', name='Original Site',
    fig.add_trace(go.Scatter(x=data_site['Timestamp'], y=data_site[feature_column], mode='markers', name='Original Site',
                             line=dict(color='dodgerblue', width=0.2, dash='dot'), opacity=0.8), row=1, col=1)

    ## Adjust the layout for a better/consistent look and display the plot
    fig.update_layout(height=800, title_text=f'[{method_description}] Original vs Imputed Data Plot of {feature_column} at {site}',
                        xaxis_title="Timestamp", yaxis_title=feature_column, 
                        xaxis=dict(
                            title="Timestamp",
                            tickmode='auto',
                            tickformat="%Y-Q%q",  ## Format: Year - Q<Quarter>
                            dtick="M3",           ## Set ticks every 3 months to indicate quarters
                        ), legend=dict(orientation="h",  ## Horizontal legend
                            yanchor="bottom", y=1.02,   ## Position legend above the plot
                            xanchor="right", x=1,       ## Align legend to the right
                            title='Legend:',            ## Optional legend title
                            bgcolor='rgba(255,255,255,0.3)',  ## Semi-transparent white background
                            bordercolor="lightgrey", borderwidth=1))
    fig.show()
 

## Instantiate the class with rawdata
imputer = DataImputer()

nz_holidays = holidays.country_holidays('NZ')
# nz_holidays = holidays.country_holidays('NZ', subdiv='AUK')

## List of imputation methods to apply, directly referencing the methods of DataImputer as tuples of (method_function, method_description)
imputation_methods = [
    # (DataImputer.mean_mode_imputation, "Mean/Mode Imputation"),
    # (lambda data: DataImputer.mean_mode_imputation(data, columns=numerical_columns), "SimpleImputer Mean/Mode Imputation"),
    (lambda data: DataImputer.mean_mode_imputation(data), "SimpleImputer Mean/Mode Imputation"),
    (DataImputer.forward_backward_imputation, "Forward/Backward Imputation"),
    (lambda data: DataImputer.interpolation_imputation(data, method='linear'), "Linear Interpolation"),
    (lambda data: DataImputer.polynomial_imputation(data, method='polynomial', order=2), "Polynomial Interpolation (Order 2)"),
    # (lambda data: DataImputer.xgboost_imputation(data, target_columns=['PM2.5', 'PM10', 'NO2']), "Machine-Learning XGBoost Imputation"),
    # (lambda data: DataImputer.prophet_imputation(data, date_column='Timestamp', target_columns=['PM2.5', 'PM10', 'NO2'], holidays=nz_holidays), "Time-Series Prophet Imputation"),
    # (lambda data: DataImputer.moving_average_imputation(data, columns=timestamp_and_numerical_columns, window='168H', min_periods=1, verbose=True), "Moving Average Imputation"),
    (lambda data: DataImputer.moving_average_imputation(data), "Moving Average Imputation"),
]

## Iterate through each imputation method, apply it, and visualize the results (data vs imputed data visualization)
for method_function, method_description in imputation_methods:
    # print(f"Applying imputation method function: {method_function}")
    # print(f"Applying imputation method description: {method_description}")

    if IS_DEBUG:
        print(f"{method_description} --------------Before Imputation--------------")
        print(cleaned_data_site1.isnull().sum(axis = 0))
        print(cleaned_data_site2.isnull().sum(axis = 0))
    ## Step 1. Apply the imputation function
    imputed_cleaned_data1 = method_function(cleaned_data_site1.copy())  ## This calls the method on the DataImputer instance
    imputed_cleaned_data2 = method_function(cleaned_data_site2.copy())  ## This calls the method on the DataImputer instance
    if IS_DEBUG:
        print(type(imputed_cleaned_data1))
        print(f"{method_description} --------------After Imputation--------------")
        print(imputed_cleaned_data1.isnull().sum(axis = 0))
        print(imputed_cleaned_data2.isnull().sum(axis = 0))
    
    ## Introduce a short pause (e.g., 30 seconds) to ensure plots are fully rendered before moving to the next item
    time.sleep(60)
    
    ## Step 2. After imputation, we can now proceed with visualization or further processing
    before_and_after_values_visualization(cleaned_data_site1.copy(), imputed_cleaned_data1, method_description, numerical_columns_S1, 'Penrose', feature_column='PM2.5')
    before_and_after_values_visualization(cleaned_data_site2.copy(), imputed_cleaned_data2, method_description, numerical_columns_S2, 'Takapuna', feature_column='PM10')


# <div class="alert alert-block alert-info">
# 🎓 End of Missing Data Imputation in Time-Series
# </div>

# In[73]:


imputed_cleaned_data1.shape


# In[74]:


imputed_cleaned_data2.shape


# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Filter Outliers

# <div class="alert alert-block alert-info">
# 🎓 Proportional Winsorization Based on Data Distribution
# </div>
# 
# This approach first identifies the actual minimum and maximum values within the acceptable range (beyond the IQR bounds) and then calculates the proportion of data points beyond these winsorized limits. It's particularly useful when you want to apply winsorization directly based on the distribution of your data, adjusting only the extreme values that fall outside the calculated bounds. As a result, this approach is more precise and aligns better with the principle of winsorization—limiting the influence of extreme outliers without removing them from the dataset.

# In[76]:


from scipy.stats.mstats import winsorize
IS_WINSORIZING_OUTLIERS = True

def calculate_iqr_bounds(data, column):
    """
    Calculate the Interquartile Range (IQR) bounds for a given column in a DataFrame.

    Args:
    - data (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column name for which to calculate the IQR bounds.

    Returns:
    - tuple: A tuple containing the lower and upper bounds.
    """
    ## Calculate IQR
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    ## Determine bounds for Winsorization based on IQR: the limits for potential outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound, Q1, Q3, IQR


# def detect_and_winsorize_outliers(data, numerical_columns):
def detect_and_handle_outliers(data, numerical_columns):
    """
    Detects outliers in numerical columns of a DataFrame using the IQR method and handles them using Winsorization.
    e.g. Apply the IQR method to detect outliers & Winsorization to handle/normalize outliers

    Limiting the influence of extreme outliers without removing them from the dataset.

    Args:
    - data (pd.DataFrame): The DataFrame to process.
    - numerical_columns (list): A list of column names to check for and handle outliers.

    Returns:
    - pd.DataFrame: A DataFrame with the outliers handled.
    """
    ## Initialize an empty list to store summary statistics
    outlier_summaries = []
    for col in numerical_columns:
        lower_bound, upper_bound, Q1, Q3, IQR = calculate_iqr_bounds(data, col)

        ## Calculate_percentiles_for_winsorization
        lower_winsorize_limit = data[col][data[col] < lower_bound].count() / data[col].count()
        upper_winsorize_limit = data[col][data[col] > upper_bound].count() / data[col].count()

        ## Collecting and calculating outlier metrics for summary
        # outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
        # num_outliers = outlier_mask.sum()
        ## Consider only non-NaN values for total records
        # num_outliers = outlier_mask.notnull().sum()

        ## Determine the percentages of the data that are outliers; this is for Winsorization limits
        num_lower_outliers = data[col][data[col] < lower_bound].count()
        num_upper_outliers = data[col][data[col] > upper_bound].count()
        num_outliers = num_lower_outliers + num_upper_outliers
        lower_outlier_percentage = num_lower_outliers / num_outliers
        upper_outlier_percentage = num_upper_outliers / num_outliers

        percentage_outliers = num_outliers / len(data) * 100

        if IS_WINSORIZING_OUTLIERS:
            # Calculate the actual min and max within the acceptable range
            valid_min = data[col][(data[col] >= lower_bound) & (data[col] <= upper_bound)].min()
            valid_max = data[col][(data[col] >= lower_bound) & (data[col] <= upper_bound)].max()
    
            ## Calculate the fractions for Winsorization
            lower_winsorize_limit = data[col][data[col] < valid_min].count() / len(data[col])
            upper_winsorize_limit = data[col][data[col] > valid_max].count() / len(data[col])
            
            ## Apply Winsorization
            # data[col] = winsorize(data[col], limits=(lower_winsorize_limit, upper_winsorize_limit))
            data[col] = winsorize(data[col], limits=(0.013653, 0.030128))

        # if IS_WINSORIZING_OUTLIERS:
            ## Instead of removing the outliers, we apply Winsorization (Winsorize column data)
            ## Winsorizing the data such that values beyond the limits are capped: Winsorize data points beyond the bounds
            # data[col] = winsorize(data[col], limits=(lower_outlier_percentage, upper_outlier_percentage))
            # data[col] = winsorize(data[col], limits=[lower_bound, upper_bound])
            # data[col] = winsorize(data[col], limits=(lower_winsorize_limit, upper_winsorize_limit))
    
            ## Applying winsorization to the column
            ## Note: winsorize operates on arrays, hence .values is used. It also modifies data in-place.
            # winsorized_data = winsorize(data[col].values, limits=(lower_winsorize_limit, upper_winsorize_limit))
            # ## Updating the column in the DataFrame with the winsorized data
            # data[col] = winsorized_data
    
            ## Apply Winsorization using calculated limits
            ## Note: It's crucial to ensure the limits are not NaN, indicating no outliers were found
            # if pd.notnull(lower_winsorize_limit) and pd.notnull(upper_winsorize_limit):
            #     data[col] = winsorize(data[col], limits=[lower_winsorize_limit, upper_winsorize_limit])

        ## Collect Outlier Summary statistics
        outlier_summaries.append({
            'Column': col,
            'Lower Winsorization Limit': lower_winsorize_limit,
            'Upper Winsorization Limit': upper_winsorize_limit,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Number of Outliers': num_outliers,
            'Percentage of Outliers': percentage_outliers,
            'No. of Lower Outliers': num_lower_outliers,
            'No. of Upper Outliers': num_upper_outliers,
            'Lower Outlier %': num_lower_outliers / data[col].notnull().sum(),
            'Upper Outlier %': num_upper_outliers / data[col].notnull().sum()
            # 'Lower Outlier %': lower_outlier_percentage * 100, ## Convert to percentage
            # 'Upper Outlier %': upper_outlier_percentage * 100  ## Convert to percentage
        })

    outlier_summary_df = pd.DataFrame(outlier_summaries)

    return data, outlier_summary_df


# In[77]:


## Detect and Handle/Retrieve Outliers
winsorized_imputed_cleaned_data1, outlier_summary_df1 = detect_and_handle_outliers(imputed_cleaned_data1.copy(), numerical_columns_S1)
winsorized_imputed_cleaned_data2, outlier_summary_df2 = detect_and_handle_outliers(imputed_cleaned_data2.copy(), numerical_columns_S2)

## Display the outlier summary
print("Outlier Summary for Site 1 (Penrose):\n")
outlier_summary_df1

## The result is a DataFrame 'rawdata' with outliers normalized/adjusted based on IQR and Winsorization
# winsorized_imputed_cleaned_data1.head()
# winsorized_imputed_cleaned_data2.head()

## TODO
## After applying IQR and Winsorization, the data is now ready for further processing such as PCA
## The `preprocessed_data` is ready for further analysis like PCA


# In[78]:


## Display the outlier summary
print("Outlier Summary or Site 2 (Takapuna):\n")
outlier_summary_df2


# In[79]:


## TODO

## Identify Outliers with extreme values
## Let's assume Air_temp > 40°C, Rel_humidity > 100, and PM2.5 > 100 are outliers
extreme_outliers = rawdata[(rawdata['Air_Temp'] > 40) | (rawdata['Rel_Humidity'] > 100) | (rawdata['PM2.5'] > 100)]

## Print information about missing data and outliers
extreme_outliers

## FIXME: winsorize(cleaned_data[FEATURE], (lower_outlier_percentage,upper_outlier_percentage))

# ## Winsorization to handle outliers
# cleaned_data['Air_Temp'] = winsorize(cleaned_data['Air_Temp'], (0.05, 0.05))
# cleaned_data['Rel_Humidity'] = winsorize(cleaned_data['Rel_Humidity'], (0.05, 0.05))
# cleaned_data['PM2.5'] = winsorize(cleaned_data['PM2.5'], (0.05, 0.05))
# cleaned_data['PM10'] = winsorize(cleaned_data['PM10'], (0.05, 0.05))

# ## Show first 5 rows of cleaned_data
# cleaned_data.head()


# In[80]:


## [What If] Visualize the difference between before and after the Winsorization
before_and_after_values_visualization(imputed_cleaned_data1, winsorized_imputed_cleaned_data1, '[What-If] Winsorization', numerical_columns_S1, 'Penrose', feature_column='PM2.5')


# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## 💾 Save Cleaned-Data to *.csv

# In[75]:


## Filter cleaned data for the specified sites
# imputed_cleaned_data1 = cleaned_data[cleaned_data['Site'] == 'Penrose']
# imputed_cleaned_data2 = cleaned_data[cleaned_data['Site'] == 'Takapuna']

## Remove redundant columns before saving to *.csv
# imputed_cleaned_data1 = cleaned_data_site1.drop(['SO2', 'Site', 'Site_Class'], axis=1)
# imputed_cleaned_data2 = cleaned_data_site2.drop(['Site', 'Site_Class'], axis=1)

# imputed_cleaned_data1
# imputed_cleaned_data1.columns
# imputed_cleaned_data2
# imputed_cleaned_data2.columns

imputed_cleaned_data1.to_csv(f"{PATH}/../cleaned_Penrose7-07May2020-to-30Apr2022.csv", index=False)
imputed_cleaned_data2.to_csv(f"{PATH}/../cleaned_Takapuna23-07May2020-to-30Apr2022.csv", index=False)


# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # 📊 Exploratory Data Analysis (EDA)

# <div class="alert alert-block alert-info">
# 🎓 In the section, Plot the data and try to extract some knowledge.</p>
# </div>

# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## 🧩 Interactive Data Visualization

# <div class="alert alert-block alert-info">
# 🎓 In the section, ....</p>
# </div>

# In[81]:


import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns


# In[82]:


## Filter rawdata for the specified sites
## Copy the rawdata for safety so the original data is not modified
pd_sites = rawdata.copy()

# ## Set 'Site' and 'Timestamp' as a multi-level index
# pd_sites.set_index(['Site', 'Timestamp'], inplace=True)
# ## TODO: log-scale ?
# # data_site1 = rawdata.loc[(site1, slice(None)), numerical_columns]
# pd_site1 = pd_sites.xs('Penrose', level='Site')[numerical_columns]
# pd_site2 = pd_sites.xs('Takapuna', level='Site')[numerical_columns]


# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Plotly Line/Bar Chart

# <div class="alert alert-block alert-info">
# 🎓 Single chart for each site with multiple features overlaid. This design simplifies comparison across different air quality metrics within the same geographical context..</p>
# </div>

# In[83]:


def plot_selected_features_plotly(data, sites, numerical_columns, selected_features):
    """
    Plots selected numerical features for specified sites using Plotly, allowing toggling visibility of features,
    with dynamic adjustment of plot height based on the number of legend items.
    
    Parameters:
    - data (DataFrame): The dataset containing 'Timestamp', 'Site', and numerical columns.
    - sites (list): List of sites to be visualized.
    - numerical_columns (list): List of all numerical columns available for plotting.
    - selected_features (list): List of features to be initially visible on the plot.
    
    Returns:
    - Plotly figure with the ability to toggle visibility of non-selected features and dynamically adjusted height.
    
    """

    ## Create an empty list to store traces
    traces = []

    ## Iterate through each site
    for site in sites:
        ## Filter the data for the current site
        site_data = data[data['Site'] == site]

        ## Iterate through each numerical feature
        for feature in numerical_columns:
            # Check if the feature should be initially visible
            is_visible = feature in selected_features

            ## Create a scatter plot for the current feature
            trace = go.Scatter(
                x=site_data['Timestamp'],
                y=site_data[feature],
                mode='lines+markers',
                name=f"{site} - {feature}",
                text=site_data[feature],
                visible='legendonly' if not is_visible else True  ## Toggle visibility based on selection
            )
            traces.append(trace)
    
    ## Calculate the dynamic height: base height + additional space per legend item
    base_height = 400
    additional_height_per_item = 20
    total_legend_items = len(sites) * len(numerical_columns)
    dynamic_height = base_height + (additional_height_per_item * total_legend_items)
    
    ## Define the layout of the plot
    layout = go.Layout(
        title='Selected Features Visualization Across Sites',
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Measurement Values'),
        legend=dict(title='Site - Feature'),
        hovermode='closest',
        height=dynamic_height  # Use the dynamically calculated height
    )
    
    ## Create the figure with traces and layout
    fig = go.Figure(data=traces, layout=layout)
    
    ## Enhance the figure by customizing axis labels, titles, and enabling the legend
    fig.update_layout(title_text='Selected Air Quality Features Across Sites')
    
    return fig


# In[84]:


sites = ['Penrose', 'Takapuna']
numerical_columns = ['AQI', 'PM10', 'PM2.5', 'SO2', 'NO', 'NO2', 'NOx', 'Wind_Speed', 'Wind_Dir', 'Air_Temp', 'Rel_Humidity']
selected_features = ['PM2.5']  # Initially visible features
fig = plot_selected_features_plotly(pd_sites, sites, numerical_columns, selected_features)

## Show the plot
fig.show()


# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Plotly MapBox Chart

# In[85]:


# def plot_scatter_mapbox(data, site_lat_dict, site_lon_dict, color_col, size_col, hover_col, title="Map View",
#                         zoom=3, height=600, width=1000,
#                         color_continuous_scale=px.colors.cyclical.IceFire, size_max=15):
#     """
#     Plots an interactive scatter map with Plotly Express using Sites' latitude and longitude.
    
#     Parameters:
#     - data: Dataframe with the data points.
#     - site_lat_dict: Dictionary mapping Sites to their latitude.
#     - site_lon_dict: Dictionary mapping Sites to their longitude.
#     - color_col: Column name for the color dimension.
#     - size_col: Column name for the size dimension.
#     - hover_col: Column name to be displayed on hover.
#     - title: Title of the map.
#     - zoom: Initial zoom level of the map.
#     - height: Height of the map.
#     - width: Width of the map.
#     - color_continuous_scale: Color scale for the map.
#     - size_max: Maximum size of data points.
#     """
    
#     # Copy to avoid modifying original data
#     plot_data = data.copy()
    
#     # Assign latitude and longitude based on Site
#     plot_data['latitude'] = plot_data['Site'].map(site_lat_dict)
#     plot_data['longitude'] = plot_data['Site'].map(site_lon_dict)
    
#     # Drop rows where latitude or longitude could not be assigned
#     plot_data.dropna(subset=['latitude', 'longitude'], inplace=True)
    
#     fig = px.scatter_mapbox(
#         plot_data,
#         lat='latitude',
#         lon='longitude',
#         color=color_col,
#         size=size_col,
#         color_continuous_scale=color_continuous_scale,
#         hover_name=hover_col,
#         height=height,
#         width=width,
#         size_max=size_max
#     )
    
#     fig.update_layout(
#         mapbox_style='open-street-map',
#         title=title,
#         hovermode='closest',
#         mapbox=dict(
#             bearing=0,
#             center=go.layout.mapbox.Center(lat=plot_data['latitude'].mean(), lon=plot_data['longitude'].mean()),
#             pitch=0,
#             zoom=zoom
#         )
#     )
#     return fig


# In[86]:


get_ipython().run_cell_magic('capture', '', '# !pip install folium\n')


# In[92]:


import folium
from folium.plugins import MarkerCluster
from IPython.display import display

## Load the dataset
data = pd.read_csv('data/extra/Air-Quality-Monitoring-Network.csv')

# Create a Folium map centered around Auckland's coordinates
# _map = folium.Map(location=[-36.848461, 174.763336], zoom_start=10)
_map = folium.Map(location=[-36.9706, 174.83834], zoom_start=10)
# _map = folium.Map(location=[-36.9706, 174.83834], zoom_start=10)

## Initialize a marker cluster for the 'cloud' icon with 'blue' color
green_marker_cluster = MarkerCluster(name='Meteorological & NO PM2.5').add_to(_map)

## Add markers for each station based on monitored pollutants and meteorological parameters
## Loop through the dataset and add markers to the map based on the data availability
for index, row in data.iterrows():
    ## Define the icon based on PM2.5 and Meteorological parameters being measured
    # icon = 'cloud' if row['Meteorological Measured'] == 'NO' else 'off' if row['PM2.5 Pollutant Monitored'] == 'NO' else 'anchor'
    # color = 'blue' if row['PM2.5 Pollutant Monitored'] == 'NO' else 'red' if row['Meteorological Measured'] == 'NO' else 'green'
    if row['Meteorological  Measured'] == 'NO':
        icon = folium.Icon(icon="cloud")
    elif row['PM2.5  Monitored'] == 'NO':
        icon = folium.Icon(color='green', icon='adjust')
    else:
        icon = folium.Icon(color='orange', prefix='fa', icon='anchor')
    
    ## Construct tooltip and popup messages with more descriptive content
    tooltip = f"{row['Site Name']} ({row['Site Class']}) Air Quality Station"
    popup = folium.Popup(f"<strong>Station Name:</strong> {row['Site Name']}<br>"
                         f"<strong>Monitored Pollutants:</strong> {row['Pollutants Monitored']}<br>"
                         f"<strong>Meteorological Measured :</strong> {row['Meteorological Parameters Measured ']}<br>"
                         f"<strong>Site Class:</strong> {row['Site Class']}<br>"
                         f"<strong>Established:</strong> {row['Established Date']}", 
                         max_width=450)
    
    marker = folium.Marker(
        location=[row['latitude'], row['longitude']],
        tooltip=tooltip,
        popup=popup,
        icon=icon,
    )
    
    ## Add the marker to the map or cluster
    if row['PM2.5  Monitored'] == 'NO':
        marker.add_to(green_marker_cluster)
    else:
        marker.add_to(_map)

## Adding a custom Legend
legend_html = '''
     <div style="position: fixed; 
     bottom: 30px; left: 30px; width: 200px; height: 90px; 
     border:1px solid grey; z-index:9999; font-size:14px;
     ">&nbsp; <b>Legend</b> <br>
     &nbsp; <i class="fa fa-cloud" style="color:blue"></i> NO Meteo.<br>
     &nbsp; <i class="fa fa-adjust" style="color:green"></i> NO PM2.5<br>
     &nbsp; <i class="fa fa-anchor" style="color:orange"></i> PM2.5 & Meteo.
      </div>
     '''
_map.get_root().html.add_child(folium.Element(legend_html))

## Save the map to an HTML file
_map.save("data/report/Air-Quality-Monitoring-Network.html")

## Display the map if running in an IPython environment, e.g., Jupyter Notebook
display(_map)


# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Plotly Gauge Chart

# In[93]:


def create_gauge_chart(data, gauge_column, gauge_column_description="Temperature (°C)", title="Temperature Status", height=400):
    """
    Creates a temperature gauge chart using Plotly's go.Indicator.

    Parameters:
    - data (DataFrame): Pandas DataFrame containing temperature data.
    - gauge_column (str): Column name in `data` that contains temperature values.
    - title (str): Title of the gauge chart.
    - height (int): Height of the figure in pixels.

    Returns:
    - A Plotly figure object displaying the temperature gauge.
    """
    
    ## Extracting the maximum, minimum, and average temperatures from the specified column
    gauge_max = data[gauge_column].max()
    gauge_min = data[gauge_column].min()
    gauge_mean = data[gauge_column].mean()
    
    ## Define the steps for the gauge based on temperature ranges with corresponding colors
    steps = [
        {'range': [gauge_min, 10], 'color': "seashell"}, ## Define seashell color ranges for the gauge
        {'range': [10, 20], 'color': "lightblue"},       ## Define lightblue color ranges for the gauge
        {'range': [20, 30], 'color': "sandybrown"},      ## Define sandybrown color ranges for the gauge
        {'range': [30, gauge_max], 'color': "tomato"}    ## Define tomato color ranges for the gauge
    ]
    
    ## Define the threshold for the gauge
    threshold = {
        'line': {'color': 'red', 'width': 4},  ## Define the threshold line's appearance
        'thickness': 0.75,                     ## Set the thickness of the threshold line
        'value': gauge_mean                    ## Set the value where the threshold line is located
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",                 ## Set the chart mode to gauge with a number
        value=gauge_mean,              ## Display the mean temperature
        domain={'x': [0, 1], 'y': [0, 1]},   ## Set the position of the gauge
        title={'text': gauge_column_description},  ## Set the title of the gauge chart
        gauge={
            'axis': {'range': [gauge_min, gauge_max]},  ## Axis range of the gauge
            'steps': steps,         ## Steps within the gauge
            'threshold': threshold  ## Threshold indicator on the gauge
        }
    ))
    
    ## Update the layout of the figure
    fig.update_layout(
        title={'text': title},  ## Title of the chart
        height=height                 ## Height of the figure
    )
    
    # fig.show()
    return fig


# In[96]:


fig = create_gauge_chart(imputed_cleaned_data1, 'Air_Temp', 'Temperature (°C)', '[Penrose] Temperature Status')
fig.show()

fig = create_gauge_chart(imputed_cleaned_data1, 'AQI', 'AQI', '[Penrose] Air Quality Index')
fig.show()

fig = create_gauge_chart(imputed_cleaned_data2, 'Air_Temp', 'Temperature (°C)', '[Takapuna] Temperature Status')
fig.show()

fig = create_gauge_chart(imputed_cleaned_data2, 'AQI', 'AQI', '[Takapuna] Air Quality Index')
fig.show()


# In[97]:


import geopandas as gpd
import pyproj

## The nz-police-district-boundaries.shx file for New Zealand police district was downloaded from Koordinates in 2021.
## This is a collection of files that map the boundaries of the 12 police districts in New Zealand.
## The original dataset is no longer available on Koordinates. These files are unaltered from the original dataset.
    
## Set up the file path and read the shapefile data
fp = "data/extra/nz-police-district-boundaries.shx" 
map_df = gpd.read_file(fp)
## Ensure the correct EPSG code
map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

## Read the csv data
df = pd.read_csv('data/extra/District.csv')

## See what the map looks like
# map_df.plot(figsize=(20, 10))

## map_df merge to df
## Rename columns and drop unnecessary ones
map_df = map_df.rename({'DISTRICT_N': 'District'}, axis='columns')
map_df = map_df.drop(columns='DISTRICT_I')
map_df = map_df.replace(['Counties/Manukau', 'Northland'], ['Counties Manukau', 'Northern'])

## Merge map_df with df, ensure that 'District' is the key for merging
df_merged = map_df.merge(df, on='District')
# df_merged = map_df.merge(df, left_on=['District'], right_on=['District'])

## Visualise with Plotly
fig = px.choropleth(df_merged, geojson=df_merged.geometry, 
                    locations=df_merged.index, color="Sum",
                    height=500,
                    color_continuous_scale="Viridis")

# Adjust map settings
fig.update_geos(fitbounds="locations", visible=True)
fig.update_layout(title_text='Map')
fig.update(layout=dict(title=dict(x=0.5)))
fig.update_layout(margin={"r":0, "t":30, "l":10, "b":10},
                  coloraxis_colorbar={'title':'Sum'})

## Show the figure
fig.show()


# # References

# * Clemente, F., Gonçalo Martins Ribeiro, Alexandre Quemy, Miriam Seoane Santos, Ricardo Cardoso Pereira, & Barros, A. (2023). ydata-profiling: Accelerating data-centric AI with high-quality data. Neurocomputing, 554, 126585–126585. https://doi.org/10.1016/j.neucom.2023.126585
# * 

# <footer style="padding-bottom:35px; background:#f9f9f9; border-bottom:3px solid #00b2b1">
#     <div style="float:left;margin-top:14px;color:#E37C4D">🎓 Predicting Air Particulate Matter at Scale ⛅️</div>
#     <div style="float:right;">
#         <div style="float:left; margin-top:14px">
#             🧑‍🎓 Auckland University of Technology (AUT)
#         </div>
#     </div>
# </footer>
