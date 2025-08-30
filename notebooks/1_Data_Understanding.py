#!/usr/bin/env python
# coding: utf-8

# <header style="padding:1px;background:#f9f9f9;border-top:3px solid #00b2b1"><img id="Teradata-logo" src="https://www.teradata.com/Teradata/Images/Rebrand/Teradata_logo-two_color.png" alt="Teradata" width="220" align="right" />
# 
# <b style='font-size:28px;font-family:Arial;color:#E37C4D'>🎓 Predicting Air Particulate Matter at Scale ⛅️</b><br>
# <b style='font-size:28px;font-family:Arial;color:#E37C4D'>🛠️ 1. Data Understanding & Preparation</b>
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
# 🎓 This section executes all of the cells in `Data_Loading_and_Descriptive_Statistics.ipynb`.
# </div>

# In[17]:


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


# In[2]:


print("\n🎓 [Site1 - Penrose]  Summary Statistics of the {site1} rawdata_site1 Dataframe such as the mean, max/minimum values ...")
rawdata_site1.describe()


# In[3]:


print("\n🎓 [Site2 - Takapuna]  Summary Statistics of the {site2} rawdata_site2 Dataframe such as the mean, maximum and minimum values ...")
rawdata_site2.describe()


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## Python Reusable Functions

# <div class="alert alert-block alert-info">
# 🎓 This section examine the key statistics and missing data percentages from the reports to gain a better understanding of the data characteristics and time-series features.
# </div>

# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # 📈 Data Understanding & Visualization

# <div class="alert alert-block alert-info">
# 🎓 This section identify and visualise Missing Data values ...
# </div>

# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Identify and Visualise Missing Data Values

# In[4]:


import missingno as msno
import matplotlib.dates as mdates


# In[5]:


def display_side_by_side_missing_data_visualizations(rawdata, site1, site2, visualization_type='matrix', feature_column='PM10'):
    """
    Displays side by side missing data visualizations for two specified sites using missingno library,
    adjusting parameters based on the visualization type.
    
    * 'matrix':  Each matrix cell color is based on whether the data exists (dark/green color) or not (white). 
                 The matrix plot is a great tool if you are working with depth-related data or time-series data. 
    * 'bar':     Displays the number of missing values for each column represented as a bar chart. 
                 It’s similar to the matrix plot but simpler to interpret.

    * 'heatmap': Displays a heat map that shows a nullity correlation between variables. The differentiates from a regular heatmap is that it measures a nullity correlation.
                 Nullity correlation (fancyimpute library) is an indicator of how strongly the absence of a variable to another variable is.
                 Values close to positive 1 indicate that the presence of null values in one column is correlated with the presence of null values in another column.
                 Values close to 0, indicate there is little to no relationship between the presence of null values in one column compared to another.

    * 'dendrogram': To show the similarity between each variable to another in terms of missing values.
                 More fully correlate variable completion, revealing trends deeper than the pairwise ones visible in the correlation heatmap
                 Hierarchical clustering algorithm calculates the similarity between variables. 
                 It has the leaves that are represented by variables and the branch that connects all variables one to the other.
                 
    Parameters:
        rawdata (DataFrame): The complete dataset containing multiple sites.
        site1 (str): The first site to display.
        site2 (str): The second site to display.
        visualization_type (str): Type of visualization. Options: 'matrix', 'bar', 'heatmap', 'dendrogram'.
    """
    
    ## Filter rawdata for the specified sites
    # data_site1 = rawdata[rawdata['Site'] == site1]
    # data_site2 = rawdata[rawdata['Site'] == site2]
    data_site1 = rawdata_site1.copy()
    data_site2 = rawdata_site2.copy()

    ## Setting up the figure to hold two subplots side by side
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharey=True)
    plt.title(f'[{visualization_type.capitalize()}] Missing Values Plot', fontsize = 18)

    ## Generate visualizations for each site & Apply visualization with correct parameters.
    if visualization_type == 'matrix':
        msno.matrix(data_site1, ax=ax[0], sparkline=False, color=(0.24, 0.77, 0.77))
        msno.matrix(data_site2, ax=ax[1], sparkline=False, color=(0.24, 0.77, 0.77))
    elif visualization_type == 'bar':
        msno.bar(data_site1, ax=ax[0], color=(0.24, 0.77, 0.77))
        msno.bar(data_site2, ax=ax[1], color=(0.24, 0.77, 0.77))
    elif visualization_type == 'heatmap':
        if data_site1.isnull().sum().sum() > 0:  ## Ensure there's at least one missing value
            msno.heatmap(data_site1, ax=ax[0])
        if data_site2.isnull().sum().sum() > 0:
            msno.heatmap(data_site2, ax=ax[1])
    elif visualization_type == 'dendrogram':
        msno.dendrogram(data_site1, ax=ax[0], orientation='right')
        msno.dendrogram(data_site2, ax=ax[1], orientation='right')
    else:
        ## Ensure the visualization type is valid
        raise ValueError("Invalid visualization_type. Choose from 'matrix', 'bar', 'heatmap', 'dendrogram'.")

    ## Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


# In[6]:


# Display side by side the missing data matrices for two specific sites, "Penrose" and "Takapuna"
display_side_by_side_missing_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='matrix')
print("The columns CO and Solar_Rad show large portions of missing data. This was identified in the bar plot, but the added benefit is you can view how that missing data is distributed in the dataframe.")

display_side_by_side_missing_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='bar')
# display_side_by_side_missing_data_visualizations(rawdata, 'Takapuna', 'Takapuna', visualization_type='heatmap')
display_side_by_side_missing_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='dendrogram')


# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Time-Series Data Visualizations

# In[7]:


def display_side_by_side_timeseries_data_visualizations(rawdata, site1, site2, visualization_type='plot', feature_column='PM10'):
    """
    Displays side by side data visualizations for two specified sites, adjusting parameters based on the visualization type.
    
    * 'plot':  
                 
    Parameters:
        rawdata (DataFrame): The complete dataset containing multiple sites.
        site1 (str): The first site to display.
        site2 (str): The second site to display.
        visualization_type (str): Type of visualization. Options: 'plot'.
    """
    
    ## Filter rawdata for the specified sites
    ## Copy the rawdata for safety so the original data is not modified
    series = rawdata.copy()
    # ## Set 'Site' and 'Timestamp' as a multi-level index
    # series.set_index(['Site', 'Timestamp'], inplace=True)
    # ## TODO: log-scale ?
    # # data_site1 = rawdata.loc[(site1, slice(None)), numerical_columns]
    # data_site1 = series.xs(site1, level='Site')[numerical_columns]
    # data_site2 = series.xs(site2, level='Site')[numerical_columns]
    
    ## Setting up the figure to hold two subplots side by side
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharey=True)
    
    # major_locator = mdates.YearLocator()
    # major_formatter = mdates.DateFormatter('%Y')
    # minor_locator = mdates.MonthLocator(bymonth=(1, 4, 7, 10))  ## Ticks for every quarter
    # minor_formatter = mdates.DateFormatter('%m')                ## [DEBUG] '%b'

    ## Generate visualizations for each site & Apply visualization with correct parameters.
    if visualization_type == 'plot':
        data_site1 = rawdata_site1.copy()[numerical_columns_S1]
        data_site2 = rawdata_site2.copy()[numerical_columns_S2]
        ## Filter rawdata for the specified sites and only include numerical columns
        data_site1.plot(ax=ax[0], title=f'[{visualization_type.capitalize()}] Data Plot for {site1}', fontsize = 18)
        data_site2.plot(ax=ax[1], title=f'[{visualization_type.capitalize()}] Data Plot for {site2}', fontsize = 18)
    elif visualization_type == 'plot_feature_column':
        ## Filter rawdata for the specified sites and only plotting the feature column
        # data_site1 = series.loc[(site1, ), feature_column]
        # data_site2 = series.loc[(site2, ), feature_column]
        # data_site1 = series.xs(site1, level='Site')[feature_column]
        # data_site2 = series.xs(site2, level='Site')[feature_column]
        data_site1 = rawdata_site1.copy()[feature_column]
        data_site2 = rawdata_site2.copy()[feature_column]
        
        data_site1.plot(ax=ax[0], title=f'[{visualization_type.capitalize()}] Data Plot of {feature_column} at {site1}', fontsize = 18, ylabel=f'{feature_column}', marker='o', linestyle='-', markersize=1)
        data_site2.plot(ax=ax[1], title=f'[{visualization_type.capitalize()}] Data Plot of {feature_column} at {site2}', fontsize = 18, ylabel=f'{feature_column}', marker='o', linestyle='-', markersize=1)
    elif visualization_type == 'plot_feature_column_1':
        ## Filter rawdata for the specified sites and only plotting the feature column
        data_site1 = rawdata_site1.copy()[feature_column]
        
        data_site1.plot(ax=ax[0], title=f'[{visualization_type.capitalize()}] Data Plot of {feature_column} at {site1}', fontsize = 18, ylabel=f'{feature_column}', marker='o', linestyle='-', markersize=1)
    else:
        ## Ensure the visualization type is valid
        raise ValueError("Invalid visualization_type. Choose from 'plot', '', '', ''.")

    ## Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


# In[8]:


display_side_by_side_timeseries_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='plot')

display_side_by_side_timeseries_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='plot_feature_column')
display_side_by_side_timeseries_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='plot_feature_column', feature_column='PM2.5')
display_side_by_side_timeseries_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='plot_feature_column', feature_column='AQI')
display_side_by_side_timeseries_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='plot_feature_column_1', feature_column='SO2')
display_side_by_side_timeseries_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='plot_feature_column', feature_column='NO')
display_side_by_side_timeseries_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='plot_feature_column', feature_column='NO2')
display_side_by_side_timeseries_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='plot_feature_column', feature_column='NOx')
display_side_by_side_timeseries_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='plot_feature_column', feature_column='Wind_Speed')
display_side_by_side_timeseries_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='plot_feature_column', feature_column='Air_Temp')
display_side_by_side_timeseries_data_visualizations(rawdata, 'Penrose', 'Takapuna', visualization_type='plot_feature_column', feature_column='Rel_Humidity')


# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Data Quality Profiling

# In[9]:


# %%capture
# !pip install llvmlite --ignore installed
# !pip install -U ydata-profiling


# In[10]:


# from ydata_profiling import ProfileReport

# ## Note: Data profiling takes more than a minute for each site !!!
# ## Setting what variables are time series: common to both sites, except those unique to one
# type_schema = {
#     # "Timestamp": "ordinal",   ## Explicitly marking the Timestamp as ordinal due to its role in indexing the data over time
#     # "Timestamp": "timeseries",
#     "AQI": "timeseries",
#     "PM10": "timeseries",
#     "PM2.5": "timeseries",
#     # "SO2": "timeseries",
#     "NO": "timeseries",
#     "NO2": "timeseries",
#     "NOx": "timeseries",
#     # "CO": "timeseries",
#     "Wind_Speed": "timeseries",
#     "Wind_Dir": "timeseries",
#     "Air_Temp": "timeseries",
#     "Rel_Humidity": "timeseries",
#     "PM2.5_Lag1": "timeseries",  ## Including lagged variables as they also represent time series data
#     "PM2.5_Lag2": "timeseries",
#     "PM10_Lag1": "timeseries",
#     "PM10_Lag2": "timeseries"
# }

# ## Unique variables for Penrose and Takapuna
# unique_penrose = {"SO2": "timeseries"}   ## 'SO2' is only for Penrose
# # unique_takapuna = {"CO": "timeseries"} ## 'CO' is only for Takapuna

# ## Combine the common schema with the unique variables for each site to create site-specific schemas
# type_schema_penrose = {**type_schema, **unique_penrose}
# # type_schema_takapuna = {**common_schema, **unique_takapuna}
# type_schema_takapuna = {**type_schema}

# ## Generate the Data Quality Profiling per monitoring Site
# # for site_name, site_data in rawdata.groupby(level='Site'):
# for site_name, site_data in rawdata.groupby("Site"):

#     ## Preprocess site_data if necessary HERE ...
#     # site_data = site_data.reset_index(level='Site', drop=True)  ## Drop the 'Site' level for profiling

#     ## Check if any series within site_data is empty or has all NaN values, then Proceed with profiling if data is valid
#     if not site_data.dropna(how='all').empty:
#         ## Apply the site-specific schema
#         type_schema = type_schema_penrose if site_name == 'Penrose' else type_schema_takapuna
        
#         ## Running 1 profile per station
#         profile = ProfileReport(
#             site_data,
#             title=f"Air Quality Profiling - Site: {site_name}",
#             explorative=True,
#             # vars={"cat": {"check_completeness": False}},  ## Optional: Adjust based on profiling needs
#             tsmode=True,
#             type_schema=type_schema,
#             sortby='Timestamp',      ## Ensure the profile respects the temporal ordering
#             correlations={"auto": {"calculate": False}}, ## FIXME
#             missing_diagrams={"Heatmap": False},         ## FIXME
#             dataset={
#                 "description": "Air Quality Data Profiling for the Research Project.",
#                 "author": "Nhat Thanh, Nguyen. 🌐 Linkedin: https://www.linkedin.com/in/nnthanh",
#                 "copyright_holder": "🎓 AUT 🛠️ Teradata's ClearScape Analytics™ ⚡",
#                 "copyright_year": 2024,
#                 "url": "https://analytics-experience.pages.dev",
#             },
#             variables={
#                 "descriptions": {
#                     "PM10": "Particulate Matter (PM10) Hourly Aggregate: µg/m³ (Micrograms per cubic metre).",
#                     "PM2.5": "Particulate Matter (PM2.5) Hourly Aggregate: µg/m³ (Micrograms per cubic metre).",
#                     "SO2": "Sulfur Dioxide (SO2) Hourly Aggregate: µg/m³ (Micrograms per cubic metre)",
#                     "NO": "Nitric Oxide (NO) Hourly Aggregate: µg/m³ (Micrograms per cubic metre)",
#                     "NO2": "Nitrogen Dioxide (NO2) Hourly Aggregate: µg/m³ (Micrograms per cubic metre)",
#                     "NOx": "Nitrogen Oxide (NOx) Hourly Aggregate: µg/m³ (Micrograms per cubic metre)",
#                     "CO": "Carbon Monoxide (CO) Hourly Aggregate: µg/m³ (Micrograms per cubic metre)",
#                     "AQI": "AQI.Air Quality Index (AQI)",
#                     "Wind_Speed": "Wind Speed Hourly Aggregate: m/s (Metres per second)",
#                     "Wind_Dir": "Wind Direction Hourly Aggregate: ° (Degrees)",
#                     "Air_Temp": "Air Temperature Hourly Aggregate: °C (Celsius)",
#                     "Rel_Humidity": "Relative Humidity Hourly Aggregate: % (Percent)",
#                     "Solar_Rad": "Solar Radiation Hourly Aggregate: W/m² (Watts per square metre)",
#                 }
#             },
#         )
#     else:
#         print(f"Skipping profiling for {site_name} due to insufficient data.")

#     ## Save each profile report to an HTML file
#     profile.to_file(f"data/report/Data_Profiling_for_{site_name}.html")

#     ## Display the profile report in the Jupyter notebook
#     print(f"Displaying profile for {site_name}:")
#     profile.to_notebook_iframe()


# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # ⚙️ Data Preprocessing

# <div class="alert alert-block alert-info">
# 🎓 This section handles missing data by using function wrappers (also known as decorators) to modify and extend an existing imputation function. Additionally, filter and display potential outliers for further investigation and analysis.
# </div>

# In[11]:


## [DEBUG]
# rawdata.columns

## Remove not usefull columns
# cleaned_data = cleaned_data.drop(['CO', 'O3', 'Solar_Rad'], axis=1)


# In[12]:


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


# In[13]:


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

# In[14]:


# !pip install --upgrade prophet xgboost --user
import functools
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from prophet import Prophet


# In[15]:


class DataImputer:
    """
    Imputation methods should directly accept a DataFrame and return the imputed DataFrame
    """
    def __init__(self):
        # if not isinstance(data, pd.DataFrame):
        #     raise ValueError("Data must be a pandas DataFrame")
        pass

    @staticmethod
    def mean_mode_imputation(data):
        """
        Impute with Mean and Mode:
        Replaces missing numerical values with the MEAN and missing categorical values with the MODE.
        """
        # print("Applying Mean/Mode Imputation...")
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                data[col].fillna(data[col].mean(), inplace=True)
            # elif return_value[col].dtype.name == 'category':
            else:  ## Assuming categorical data
                data[col].fillna(data[col].mode()[0], inplace=True)
        return data

    @staticmethod
    def forward_backward_imputation(data):
        """
        Impute with Forward and Backward propagation
        Using ffill and bfill as a naive method, to complete the data.
        """
        # print("Applying Forward/Backward Imputation...")
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
    def moving_average_imputation(data):
        """
        Impute with moving average
        A moving average is better at adapting to changes by considering a few nearby data points to compute the mean.
        Yet, it can still lead to biased results if the data is not missing at random.
        """
        # print("Applying Moving Average Imputation...")
        ## Select only numeric columns for rolling operation
        numeric_cols = data.select_dtypes(include=[np.number])
        rolling_window = 52 ## A year has 52 weeks (52 weeks * 7 days per week) approximately.
        ## Apply rolling mean to numeric columns only
        numeric_rolling_mean = numeric_cols.rolling(window=rolling_window, min_periods=1).mean()
        ## Update the original data with the rolling mean values
        data.update(numeric_rolling_mean)
        return data

    @staticmethod
    def prophet_imputation(data, date_column, target_columns):
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
            ## TODO: Initialize and fit Prophet model
            model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
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


# <div class="alert alert-block alert-info">
# 🎓 [BACKUP] Missing Values Imputation using Wrapper Functions (Decorators)
# </div>

# In[19]:


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

## List of imputation methods to apply, directly referencing the methods of DataImputer as tuples of (method_function, method_description)
imputation_methods = [
    (DataImputer.mean_mode_imputation, "Mean/Mode Imputation"),
    (DataImputer.forward_backward_imputation, "Forward/Backward Imputation"),
    (DataImputer.moving_average_imputation, "Moving Average Imputation"),
    (lambda data: DataImputer.interpolation_imputation(data, method='linear'), "Linear Interpolation"),
    (lambda data: DataImputer.polynomial_imputation(data, method='polynomial', order=2), "Polynomial Interpolation (Order 2)"),
    (lambda data: DataImputer.xgboost_imputation(data, target_columns=['PM2.5', 'PM10', 'NO2']), "Machine-Learning XGBoost Imputation"),
    (lambda data: DataImputer.prophet_imputation(data, date_column='Timestamp', target_columns=['PM2.5', 'PM10', 'NO2']), "Time-Series Prophet Imputation"),
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
    # before_and_after_values_visualization(cleaned_data_site2.copy(), imputed_cleaned_data2, method_description, numerical_columns_S2, 'Takapuna', feature_column='PM10')


# <div class="alert alert-block alert-info">
# 🎓 End of Missing Data Imputation in Time-Series
# </div>

# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Filter Outliers

# <div class="alert alert-block alert-info">
# 🎓 Proportional Winsorization Based on Data Distribution
# </div>
# 
# This approach first identifies the actual minimum and maximum values within the acceptable range (beyond the IQR bounds) and then calculates the proportion of data points beyond these winsorized limits. It's particularly useful when you want to apply winsorization directly based on the distribution of your data, adjusting only the extreme values that fall outside the calculated bounds. As a result, this approach is more precise and aligns better with the principle of winsorization—limiting the influence of extreme outliers without removing them from the dataset.

# In[21]:


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


# In[24]:


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


# In[25]:


## Display the outlier summary
print("Outlier Summary or Site 2 (Takapuna):\n")
outlier_summary_df2


# In[26]:


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


# In[28]:


## [What If] Visualize the difference between before and after the Winsorization
before_and_after_values_visualization(imputed_cleaned_data1, winsorized_imputed_cleaned_data1, '[What-If] Winsorization', numerical_columns_S1, 'Penrose', feature_column='PM2.5')


# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## 💾 Save Cleaned-Data to *.csv

# In[20]:


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


# <header style="padding:1px;border-top:3px solid #E37C4D">
# 
# ## Data Quality Profiling

# In[30]:


# %%capture
# !pip install llvmlite --ignore installed
# !pip install -U ydata-profiling


# In[31]:


from ydata_profiling import ProfileReport

## Note: Data profiling takes more than a minute for each site !!!
## Setting what variables are time series: common to both sites, except those unique to one
type_schema = {
    # "Timestamp": "ordinal",   ## Explicitly marking the Timestamp as ordinal due to its role in indexing the data over time
    # "Timestamp": "timeseries",
    "AQI": "timeseries",
    "PM10": "timeseries",
    "PM2.5": "timeseries",
    # "SO2": "timeseries",
    "NO": "timeseries",
    "NO2": "timeseries",
    "NOx": "timeseries",
    # "CO": "timeseries",
    "Wind_Speed": "timeseries",
    "Wind_Dir": "timeseries",
    "Air_Temp": "timeseries",
    "Rel_Humidity": "timeseries",
    "PM2.5_Lag1": "timeseries",  ## Including lagged variables as they also represent time series data
    "PM2.5_Lag2": "timeseries",
    "PM10_Lag1": "timeseries",
    "PM10_Lag2": "timeseries"
}

## Unique variables for Penrose and Takapuna
unique_penrose = {"SO2": "timeseries"}   ## 'SO2' is only for Penrose
# unique_takapuna = {"CO": "timeseries"} ## 'CO' is only for Takapuna

## Combine the common schema with the unique variables for each site to create site-specific schemas
type_schema_penrose = {**type_schema, **unique_penrose}
# type_schema_takapuna = {**common_schema, **unique_takapuna}
type_schema_takapuna = {**type_schema}

## Generate the Data Quality Profiling per monitoring Site
# for site_name, site_data in rawdata.groupby(level='Site'):
for site_name, site_data in rawdata.groupby("Site"):

    ## Preprocess site_data if necessary HERE ...
    # site_data = site_data.reset_index(level='Site', drop=True)  ## Drop the 'Site' level for profiling

    ## Check if any series within site_data is empty or has all NaN values, then Proceed with profiling if data is valid
    if not site_data.dropna(how='all').empty:
        ## Apply the site-specific schema
        type_schema = type_schema_penrose if site_name == 'Penrose' else type_schema_takapuna
        
        ## Running 1 profile per station
        profile = ProfileReport(
            site_data,
            title=f"Air Quality Profiling - Site: {site_name}",
            explorative=True,
            # vars={"cat": {"check_completeness": False}},  ## Optional: Adjust based on profiling needs
            tsmode=True,
            type_schema=type_schema,
            sortby='Timestamp',      ## Ensure the profile respects the temporal ordering
            correlations={"auto": {"calculate": False}}, 
            missing_diagrams={"Heatmap": False},         
            dataset={
                "description": "Air Quality Data Profiling for the Research Project.",
                "author": "Nhat Thanh, Nguyen. 🌐 Linkedin: https://www.linkedin.com/in/nnthanh",
                "copyright_holder": "🎓 AUT 🛠️ Teradata's ClearScape Analytics™ ⚡",
                "copyright_year": 2024,
                "url": "https://analytics-experience.pages.dev",
            },
            variables={
                "descriptions": {
                    "PM10": "Particulate Matter (PM10) Hourly Aggregate: µg/m³ (Micrograms per cubic metre).",
                    "PM2.5": "Particulate Matter (PM2.5) Hourly Aggregate: µg/m³ (Micrograms per cubic metre).",
                    "SO2": "Sulfur Dioxide (SO2) Hourly Aggregate: µg/m³ (Micrograms per cubic metre)",
                    "NO": "Nitric Oxide (NO) Hourly Aggregate: µg/m³ (Micrograms per cubic metre)",
                    "NO2": "Nitrogen Dioxide (NO2) Hourly Aggregate: µg/m³ (Micrograms per cubic metre)",
                    "NOx": "Nitrogen Oxide (NOx) Hourly Aggregate: µg/m³ (Micrograms per cubic metre)",
                    "CO": "Carbon Monoxide (CO) Hourly Aggregate: µg/m³ (Micrograms per cubic metre)",
                    "AQI": "AQI.Air Quality Index (AQI)",
                    "Wind_Speed": "Wind Speed Hourly Aggregate: m/s (Metres per second)",
                    "Wind_Dir": "Wind Direction Hourly Aggregate: ° (Degrees)",
                    "Air_Temp": "Air Temperature Hourly Aggregate: °C (Celsius)",
                    "Rel_Humidity": "Relative Humidity Hourly Aggregate: % (Percent)",
                    "Solar_Rad": "Solar Radiation Hourly Aggregate: W/m² (Watts per square metre)",
                }
            },
        )
    else:
        print(f"Skipping profiling for {site_name} due to insufficient data.")

    ## Save each profile report to an HTML file
    profile.to_file(f"data/report/Data_Profiling_for_{site_name}.html")

    ## Display the profile report in the Jupyter notebook
    print(f"Displaying profile for {site_name}:")
    profile.to_notebook_iframe()


# # References

# * Clemente, F., Gonçalo Martins Ribeiro, Alexandre Quemy, Miriam Seoane Santos, Ricardo Cardoso Pereira, & Barros, A. (2023). ydata-profiling: Accelerating data-centric AI with high-quality data. Neurocomputing, 554, 126585–126585. https://doi.org/10.1016/j.neucom.2023.126585

# <footer style="padding-bottom:35px; background:#f9f9f9; border-bottom:3px solid #00b2b1">
#     <div style="float:left;margin-top:14px;color:#E37C4D">🎓 Predicting Air Particulate Matter at Scale ⛅️</div>
#     <div style="float:right;">
#         <div style="float:left; margin-top:14px">
#             🧑‍🎓 Auckland University of Technology (AUT)
#         </div>
#     </div>
# </footer>
