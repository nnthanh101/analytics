#!/usr/bin/env python
# coding: utf-8

# <header style="padding:1px;background:#f9f9f9;border-top:3px solid #00b2b1"><img id="Teradata-logo" src="https://www.teradata.com/Teradata/Images/Rebrand/Teradata_logo-two_color.png" alt="Teradata" width="220" align="right" />
# 
# <b style = 'font-size:28px;font-family:Arial;color:#E37C4D'>🎓 Predicting Air Particulate Matter at Scale ⛅️</br> 🛠️ 3. Time Series Analysis and Forecasting</b>
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
# 🎓 This notebook shows various methods that can be used to analyse and predict a Time Series Dataset. It should be ready for reuse in the next steps (Machine Learning / Deep Learning Modelling and Intelligent Dashboard) in CRISP-DM for Data Science.
#     
# </div>

# * **Workflow steps:**
# 
#   1. Import the required teradataml modules.
#      * Connect to a Vantage system.
#   2. 📈 Time-Series Analysis
#   3. ⚙️ Time-Series Forecasting
#   4. Cleanup.

# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
#     
# # 🎯 Libraries and Reusable Functions 

# <div class="alert alert-block alert-info">
# 🎓 This section executes all of the cells in `Data_Loading_and_Descriptive_Statistics.ipynb`.
# </div>

# In[2]:


import logging

## TODO: .env --> determines the environment for output format programmatically
## Check for the JupyterLab environment, which might affect how visualizations are rendered or interacted with.
# IS_JUPYTERLAB               = True  ## True if JupyterLab; False if Python .py
# IS_TERADATA_VANTAGE         = False ## True if Data in Teradata Vantage; False if Laptop/Virtual-Machine
# IS_DATA_IN_TERADATA_VANTAGE = False ## True if Data in Teradata Vantage; False if Data from *.csv/*xls files
IS_DEBUG                    = True  ## Plot and display additional information or not

# if not IS_DEBUG:
## Set logging level to WARNING to suppress info messages --> turn-off Prophet logs
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
# logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
# logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.WARNING)

get_ipython().run_line_magic('run', '-i ./Data_Loading_and_Descriptive_Statistics.ipynb')


# In[2]:


print("\n🎓 [Site1 - Penrose]  Summary Statistics of the {site1} rawdata_site1 Dataframe such as the mean, max/minimum values ...")
rawdata_site1.describe()


# In[3]:


print("\n🎓 [Site2 - Takapuna]  Summary Statistics of the {site2} rawdata_site2 Dataframe such as the mean, maximum and minimum values ...")
rawdata_site2.describe()


# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # 📈 Time-Series Analysis: 

# <div class="alert alert-block alert-info">
# 🎓 Time series analysis prepares and analyzes time series datasets for time series forecasting using <b>ARIMA</b> (AutoRegressive Integrated Moving Average) model. The results of time series analysis will extract useful information from time series data, such as trends, cyclic and seasonal deviations, correlations, etc.</p>
# </div>

# <header style="padding:1px;border-top:3px solid #E37C4D"></header>
# 
# <p><img style="float:right; margin:0px; padding:0px; max-height:450px" src="https://otexts.com/fpp3/figs/arimaflowchart.png"></p>
# 
# 1. [x] <font color='blue'>**check_stationarity**</font>: checking for **Stationarity of Time Series** using the **ADF Test**
# 
#     <details>
#     <summary>Checking for stationarity of Time Series using the Augmented Dickey-Fuller (ADF) Test</summary>
#         
#     * **Time-Series Stationarity**: Some time-series models, such as such as ARIMA, assume that the underlying data is stationary. Stationarity describes that the time-series has
#         * Constant mean and mean is not time-dependent
#         * Constant variance and variance is not time-dependent
#         * Constant covariance and covariance is not time-dependent
#     * The check for stationarity can be done via three different approaches:
#         * [ ] **Visually**: plot time series and check for trends or seasonality
#         * [ ] **Basic statistics**: split time series and compare the mean and variance of each partition
#         * [x] **Statistical test**: Augmented Dickey Fuller test
#     * Apply the **ADF Test** to verify if the time series data is stationary, which is a requirement for certain time series forecasting methods like ARIMA.
# 
#     * **Augmented Dickey-Fuller (ADF)** test is a type of statistical test called a unit root test. Unit roots are a cause for non-stationarity.
#     
#         * **Null Hypothesis (H0)**: Time series has a unit root. (Time series is not stationary).
#         * **Alternate Hypothesis (H1)**: Time series has no unit root (Time series is stationary).
#         ---
#         * **If the null hypothesis can be rejected, we can conclude that the time series is stationary.** There are two ways to rejects the null hypothesis:    
#             * On the one hand, the null hypothesis can be rejected if the p-value is below a set significance level. The defaults significance level is 5%
#                 * <font color='red'>**p-value > significance level (default: 0.05)**</font>: Fail to reject the null hypothesis (H0), the data has a unit root and is <font color='red'>non-stationary</font>.
#                 * <font color='green'>**p-value <= significance level (default: 0.05)**</font>: Reject the null hypothesis (H0), the data does not have a unit root and is <font color='green'>stationary</font>.
#                 
#             * On the other hand, the null hypothesis can be rejects if the test statistic is less than the critical value.
#                 * <font color='red'>**ADF statistic > critical value**</font>: Fail to reject the null hypothesis (H0), the data has a unit root and is <font color='red'>non-stationary</font>.
#                 * <font color='green'>**ADF statistic < critical value**</font>: Reject the null hypothesis (H0), the data does not have a unit root and is <font color='green'>stationary</font>.
# 
#     </details>
# 
#     <details>
#     <summary>Checking for stationarity of Time Series using the Kwiatkowski — Phillips — Schmidt — Shin (KPSS) Test</summary>
#         
#         * The KPSS test can also be used to detect stochastic trends.
#         * The test hypotheses are opposite relative to ADF:
#             * Null hypothesis: the time series is trend-stationary.
#    
#     </details>
# 
#     </br>
#     
# 1. [ ] determine_acf_pacf: determine ACF and PACF for Autocorrelation and Partial Autocorrelation of the Time Series.
#     * ACF calculates the autocorrelation or autocovariance of a time series. The autocorrelation and autocovariance show how the time series correlates or covaries with itself when delayed by a lag in time or space. 
# 
# ---
# 
# > <font color='green'>Fitting and Validating ARIMA models parameters (p, d, q) based on AIC for Penrose and Takapuna. </br> It involves determining the best-fit model and predicting future demand forecasts.</font>
# 
# 3. [x] fit: Estimate/Fit the ARIMA model parameters for forecasting pollutant levels.
# 4. [x] validate: Model Validation and Scoring: Validate the ARIMA model against a portion of the data to determine the model’s accuracy.
# 5. [ ] forecast: Forecast future pollutant levels using the validated model.
# 6. [ ] visualize actual versus predicted values and forecasted values to assess model performance.
#     * [ ] plot_diagnostics: plot diagnostic charts
#     * [ ] generate_report: generate a report of the model fit and forecast

# In[4]:


get_ipython().run_cell_magic('capture', '', '# !pip install statsmodels pmdarima\n')


# In[7]:


class AdvancedTimeSeriesAnalysis:
    """
    Enhanced Time Series Analysis focusing on effective visualization and communication of air quality data results/insights.
    Note: The data should be in chronological order and the timestamps should be equidistant in time series.

    Initialization: The initialization method configures the dataframe, ensuring that the 'Timestamp' and 'Site' columns exist and are properly formatted.

    Feature Engineering: Additional time-based features are added to the dataframe, which can be beneficial for exploring seasonal patterns and dependencies.
    
    Scaling Features: If scaling is required for certain machine learning models, this method prepares the scaled dataframe.
    
    Training and Testing Split: Methods for splitting the data into training and testing sets are implemented, with options of using random splits or a cutoff date.
    
    Seasonal Decomposition: This method performs seasonal decomposition for selected sites and visualises the components. It's critical for understanding underlying trends and seasonal patterns in air quality data.
    
    Stationarity Test: Performs ADF and KPSS tests to evaluate the stationarity of the time series, which is required for proper ARIMA modelling.
    
    Feature Importance: Visualization methods for model feature importances provide insight into which factors most influence predictions.
    
    Forecast Plotting: Plot methods for comparing forecasts with actual values and visualizing the confidence intervals around predictions.
    """
    
    def __init__(self, series, timestamp_col='Timestamp'):
        """
        Initializes the class with a dataframe and sets the timestamp column as the datetime index.
        Assumes the dataframe has a column named as specified by `timestamp_col` which should be converted to datetime if not already.
        
        Parameters:
        - dataframe: A pandas DataFrame that includes a timestamp column.
        - timestamp_col: The name of the column in `dataframe` to set as the datetime index.
        """

        # self.df = pd.read_csv(filepath, parse_dates=['Timestamp'], index_col='Timestamp')
        self.df = series.copy()
        # if timestamp_col not in self.df.columns:
        #     raise ValueError(f"The dataframe does not contain a column named '{timestamp_col}'")
        # ## Convert the timestamp column to datetime if it's not already
        # if not pd.api.types.is_datetime64_any_dtype(self.df[timestamp_col]):
        #     self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        
        ## Step 1. Prophet expects the dataset to have two columns: ds and y. 
        ##         The ds column should be of a date format and y the variable we wish to forecast.
        # series.rename(columns={'Timestamp': 'ds', 'PM10': 'y'}, inplace=True)
        self.df.rename(columns={'Timestamp': 'ds'}, inplace=True)

        ## Handle multiple sites with ordered timestamps with multi-level index (or hierarchical index)
        if 'ds' not in self.df.columns or 'Site' not in self.df.columns:
            raise ValueError("DataFrame must contain 'Timestamp' and 'Site' columns.")
        ## Sets a multi-level index using 'Timestamp'|'ds' and 'Site'. This is crucial for localized time-series analyses.
        self.df['ds'] = pd.to_datetime(self.df['ds'])
        self.df.set_index(['Site', 'ds'], inplace=True)
        
        # self.engineer_features()
        # self.scale_features()

    
    # def engineer_features(self):
    #     """
    #     Adds time-based features to the dataframe to enhance the analysis capabilities, adjusting for the multi-level index.
    #     This includes extracting various temporal components and creating lag features for PM2.5 and PM10 variables to analyze time-based dependencies.
    #     """
    #     ## Extracting date-time components
    #     timestamp_index = self.df.index.get_level_values('ds') ## 'Timestamp'
    #     # self.df['Hour']       = timestamp_index.hour
    #     self.df['Day']        = timestamp_index.day
    #     self.df['DayOfWeek']  = timestamp_index.dayofweek
    #     self.df['Month']      = timestamp_index.month
    #     self.df['Quarter']    = timestamp_index.quarter
    #     self.df['Year']       = timestamp_index.year

    #     # self.df['DayOfYear']  = timestamp_index.dayofyear
    #     ## Extract week of year for each timestamp
    #     self.df['WeekOfYear'] = [d.isocalendar()[1] for d in timestamp_index]
        
    #     ## Calculating the season based on the month --> accurately reflects the local climate and seasonal cycles
    #     ## Season encoding: 1 (Summer): December, January, February; 2 (Autumn): March, April, May
    #     ##                  3 (Winter): June, July, August         ; 4 (Spring): September, October, November
    #     ## Adjusted for meteorological seasons in Auckland, New Zealand:  
    #     # self.df['Season'] = (((timestamp_index.month % 12) + 1) // 3) % 4 + 1 
    #     ## TODO: Southern Hemisphere like New Zealand and Australia vs Northern Hemisphere like England
    #     # Correctly mapping the month to meteorological seasons for Southern Hemisphere (Auckland)
    #     self.df['Season'] = self.df['Month'].apply(
    #         lambda x: 1 if 9 <= x <= 11 else       ## Spring: Sep, Oct, Nov
    #                   2 if 12 <= x or x <= 2 else  ## Summer: Dec, Jan, Feb
    #                   3 if 3 <= x <= 5 else        ## Autumn: Mar, Apr, May
    #                   4                            ## Winter: Jun, Jul, Aug
    #     )

    #     ## Adding lag features for PM2.5 and PM10 to capture previous time steps' influence --> to compare the correlation with the other variables.
    #     self.df['PM2.5_Lag1'] = self.df.groupby(level='Site')['PM2.5'].shift(1)
    #     self.df['PM2.5_Lag2'] = self.df.groupby(level='Site')['PM2.5'].shift(2)
    #     self.df['PM10_Lag1'] = self.df.groupby(level='Site')['PM10'].shift(1)
    #     self.df['PM10_Lag2'] = self.df.groupby(level='Site')['PM10'].shift(2)
    #     if IS_DEBUG:
    #         self.df[['ds', 'Year', 'Quarter','Season', 'Month', 'Day', 'DayOfYear', 'DayOfWeek']].head()

    
    def scale_features(self):
        """
        Scales features to be used in machine learning and deep learning models.
        """
        ## FIXME
        # scaler = MinMaxScaler()
        # self.df_scaled = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns, index=self.df.index)
        self.df_scaled = self.df

    def train_test_split(self, target, shuffle=False):
        """
        Splits the data into training and testing sets.
        """
        X = self.df_scaled.drop(columns=[target])
        y = self.df_scaled[target]
        return train_test_split(X, y, test_size=0.2, shuffle=shuffle)

    def train_test_split_cutoff_date(self, cutoff_date='2022-12-31', is_debug=False):
        """
        Splits the data into training and testing sets based on a cutoff date for each site.
        Provides detailed information about the shape of each dataset and prints this information if debugging is enabled.
        """
        cutoff_date = pd.to_datetime(cutoff_date)
        training_data = {}
        testing_data = {}
        sites = self.df.index.get_level_values('Site').unique()

        ## TODO: using numerical_columns_S1, numerical_columns_S2
        ## Define the specific columns for each site if known; otherwise use all columns available
        site_columns = {
            'Penrose': ['AQI', 'PM10', 'PM2.5', 'SO2', 'NO', 'NO2', 'NOx', 'Wind_Speed', 'Wind_Dir', 'Air_Temp', 'Rel_Humidity'],
            'Takapuna': ['AQI', 'PM10', 'PM2.5', 'NO', 'NO2', 'NOx', 'Wind_Speed', 'Wind_Dir', 'Air_Temp', 'Rel_Humidity']
        }

        for site in sites:
            site_data = self.df.xs(site, level='Site')
            ## Ensure there is data for the site
            if site_data.empty:
                print(f"No data available for site: {site}. Skipping...")
                continue
            ## Filter data to include only relevant columns for the site
            relevant_columns = site_columns.get(site, site_data.columns)  ## Default to all columns if site not specified
            site_data = site_data[relevant_columns].copy()                ## Ensure only relevant columns are considered

            ## Split data based on the cutoff date
            train = site_data[site_data.index <= cutoff_date]
            test = site_data[site_data.index > cutoff_date]

            ## Store training and testing data in dictionaries
            training_data[site] = train
            testing_data[site] = test

            ## Print information about the splits
            if is_debug:
                print(f"Site: {site}")
                print(f"Series shape: {site_data.shape}, Train DataFrame shape: {train.shape}, Test DataFrame shape: {test.shape}")
                if not train.empty:
                    print(f"Training data from {train.index.min()} to {train.index.max()}")
                if not test.empty:
                    print(f"Testing data from {test.index.min()} to {test.index.max()} \n")

        return training_data, testing_data

  
    def seasonal_decompose(self, target, model='additive', extrapolate_trend='freq', period=24):
        """
        Performs and visualizes seasonal decomposition of the target variable for specified sites.
        Plots the trend, seasonal, and residual components side by side for 'Penrose' and 'Takapuna'.
        
        Parameters:
            target (str): The target variable column name for decomposition (e.g., 'PM2.5').
            model (str): The type of decomposition model ('additive' or 'multiplicative').
            period (int): The periodicity of the time series data (default is 24 for hourly data).
        """
        print(f"Seasonal Decomposition: Analyzes and plots seasonal decomposition of the {target} target variable.")
        sites = ['Penrose', 'Takapuna']
        fig, axes = plt.subplots(nrows=3, ncols=len(sites), figsize=(12, 12), sharex=True)
        
        for i, site in enumerate(sites):
            ## Extract site-specific data
            site_data = self.df.xs(site, level='Site')[target].dropna()
            if len(site_data) < 2 * period:
                raise ValueError(f"Not enough data points for seasonal decomposition at {site} site with period {period}.")

            # ## Check sufficient data for decomposition
            # if len(site_data) < 2 * period:
            #     raise ValueError(f"Not enough data points for seasonal decomposition at {site} site with period {period}.")
            ## Seasonal decomposition
            decomposition = sm.tsa.seasonal_decompose(site_data, model=model, extrapolate_trend=extrapolate_trend, period=period)
            components = [decomposition.trend, decomposition.seasonal, decomposition.resid]
            component_names = ['Trend', 'Seasonal', 'Residual']

            for j, component in enumerate(components):
                axes[j, i].plot(component, label=f'{site} {component_names[j]}')
                axes[j, i].set_title(f'{component_names[j]} for {site}')
                axes[j, i].legend()

        plt.tight_layout()
        plt.show()


    def stationarity_test(self):
        """
        Performs the Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity tests on numeric columns across different sites to assess stationarity
        and outputs detailed results in DataFrame format, including hypothesis evaluation.
        [ADF Test][Null Hypothesis H0]: The series has a unit root (non-stationary).
        [KPSS Test][Null Hypothesis H0]: The series is stationary or trend-stationary.

        
        ADF tests for unit roots, while KPSS tests for stationarity around a trend.

        Returns a structured DataFrame with test results and logs any issues with data directly to the console.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        sites = ['Penrose', 'Takapuna']
        results = []

        for site in sites:
            site_data = self.df.xs(site, level='Site')
            for column in numeric_cols:
                series_data = site_data[column].dropna()
                if series_data.empty:
                    print(f"Warning: No data available for {column} at {site} after dropna(). Skipping...")
                    continue

                ## Adjusters Dickey-Fuller (ADF) Test
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adf_test_result = adfuller(series_data, autolag='AIC')
                
                ## Kwiatkowski — Phillips — Schmidt — Shin (KPSS) Test
                ## The KPSS test can also be used to detect stochastic trends.
                ## The test hypotheses are opposite relative to ADF:
                ## Null hypothesis: the time series is trend-stationary.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kpss_test_result = kpss(series_data, regression='ct')

                ## Prepare results for both tests including hypothesis interpretation
                results.append({
                    'Site': site,
                    'Variable': column,
                    'ADF Statistic': adf_test_result[0],
                    'ADF p-value': adf_test_result[1],
                    'ADF Critical Values': adf_test_result[4],
                    'Used Lag': adf_test_result[2],
                    # 'Observations': adf_test_result[3],
                    # 'Critical Value (1%)': adf_test_result[4]['1%'],
                    'Critical Value (5%)': adf_test_result[4]['5%'],
                    # 'Critical Value (10%)': adf_test_result[4]['10%'],
                    
                    'KPSS Statistic': kpss_test_result[0],
                    'KPSS p-value': kpss_test_result[1],
                    'KPSS Critical Values': kpss_test_result[3],
                    'Lags Used': kpss_test_result[2],
                    # 'Critical Value (1%)': kpss_test_result[3]['1%'],
                    'Critical Value (5%)': kpss_test_result[3]['5%'],
                    # 'Critical Value (10%)': kpss_test_result[3]['10%'],
                    ## ADF Test: Non-stationary if test statistic is greater than or equal to the 5% critical value
                    'ADF Conclusion': "Stationary" if adf_test_result[0] < adf_test_result[4]['5%'] else "Non-stationary",
                    ## KPSS Test: Non-stationary if test statistic is greater than or equal to the 5% critical value
                    'KPSS Conclusion': "Non-stationary" if kpss_test_result[0] >= kpss_test_result[3]['5%'] else "Stationary",
                })

        ## Convert results to DataFrame for better accessibility and display
        result_df = pd.DataFrame(results)
        return result_df


    def plot_feature_importance(self, model, features):
        """
        Visualizes the importance of each feature in the model.
        """
        importance = model.feature_importances_
        fig = px.bar(x=features, y=importance, title='Feature Importance')
        fig.update_xaxes(title_text='Feature')
        fig.update_yaxes(title_text='Importance')
        fig.show()

    def feature_importance_plot(self, model, features):
        """
        Visualizes the feature importance from a fitted machine learning model.
        """
        importance = model.feature_importances_
        fig = go.Figure([go.Bar(x=features, y=importance)])
        fig.update_layout(title='Feature Importance', xaxis_title='Features', yaxis_title='Importance')
        fig.show()

    def plot_forecast_vs_actual(self, y_actual, y_pred, title='Forecast vs Actual'):
        """
        Plots actual values against predictions with an area for confidence intervals.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_actual.index, y=y_actual, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='lines', name='Predicted'))
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Value')
        fig.show()
    
    def forecast_plot(self, historical, forecast, title='Forecast vs Actuals'):
        """
        Plots historical data along with forecasts and confidence intervals if available.
        """
        historical_data = go.Scatter(x=historical.index, y=historical, mode='lines', name='Historical')
        forecast_data = go.Scatter(x=forecast.index, y=forecast['Prediction'], mode='lines', name='Forecast')
        lower_bound = go.Scatter(x=forecast.index, y=forecast['Lower CI'], mode='lines', name='Lower Confidence Interval', line=dict(color='lightgrey'), showlegend=False)
        upper_bound = go.Scatter(x=forecast.index, y=forecast['Upper CI'], mode='lines', name='Upper Confidence Interval', fill='tonexty', line=dict(color='lightgrey'), showlegend=False)
        layout = go.Layout(title=title, xaxis_title='Time', yaxis_title='Value', hovermode='closest')
        fig = go.Figure(data=[historical_data, forecast_data, lower_bound, upper_bound], layout=layout)
        fig.show()


# In[8]:


## Filter rawdata for the specified sites
## [Note] Unsure impute missing values first
## Copy the rawdata for safety so the original data or prepared data from Teradata Vantage is not modified
if IS_DATA_IN_TERADATA_VANTAGE:
    series = df_table.to_pandas()
else:
    series = rawdata.copy()

timeseries_analysis = AdvancedTimeSeriesAnalysis(series, timestamp_col='Timestamp')
## Additive model; period: 12 months in monthly dataset | daily dataset would be period=365 | hourly dataset would be 24.
print("\n[PM2.5] Plots the trend, seasonal, and residual components side by side for 'Penrose' and 'Takapuna'\n")
timeseries_analysis.seasonal_decompose('PM2.5', model='additive', extrapolate_trend='freq', period=24)
print("[PM10] Plots the trend, seasonal, and residual components side by side for 'Penrose' and 'Takapuna'")
timeseries_analysis.seasonal_decompose('PM10', model='additive', extrapolate_trend='freq', period=24)

## ADF & KPSS test results
stationarity_results = timeseries_analysis.stationarity_test()
stationarity_results


# In[9]:


training_data, testing_data = timeseries_analysis.train_test_split_cutoff_date(cutoff_date='2021-12-31', is_debug=True)

## Retrieve training data for 'Penrose' & 'Takapuna'
training_data1 = training_data['Penrose']
training_data2 = training_data['Takapuna']

## Retrieve testing data for 'Penrose' & 'Takapuna'
testing_data1 = testing_data['Penrose']
testing_data2 = testing_data['Takapuna']

print(f"Series 1 shape - Penrose: {rawdata_site1.shape}, Train DataFrame shape: {training_data1.shape}, Test DataFrame shape: {testing_data1.shape}")
print(f"Series 2 shape - Takapuna: {rawdata_site2.shape}, Train DataFrame shape: {training_data2.shape}, Test DataFrame shape: {testing_data2.shape}")

## TODO
# model, predictions = some_model_training_function(...)
# analysis.plot_forecast_vs_actual(analysis.df['PM10'], predictions)
# analysis.plot_feature_importance(model, analysis.df.columns)


# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # 📈 Time-Series Analysis & Forecasting using ARIMA

# <div class="alert alert-block alert-info">
# 🎓 Time series analysis prepares and analyzes time series datasets for time series forecasting using <b>ARIMA</b> (AutoRegressive Integrated Moving Average) model. The results of time series analysis will extract useful information from time series data, such as trends, cyclic and seasonal deviations, correlations, etc.</p>
# </div>

# In[10]:


class AdvancedTimeSeriesForecastingARIMA:

    def __init__(self, training_data, testing_data, target_variable):
        """
        Initializes the analysis with training and testing data, 
        ensuring that both datasets contain the specified target variable.
        Args:
        - training_data (DataFrame): The training dataset including the target variable.
        - testing_data (DataFrame): The testing dataset including the target variable.
        - target_variable (str): The name of the column to forecast.
        """
        if target_variable not in training_data.columns or target_variable not in testing_data.columns:
            raise ValueError(f"Both training and testing data must contain the column {target_variable}")
        
        self.training_data = training_data
        self.testing_data = testing_data
        self.target_variable = target_variable
        self.model = None
        self.forecast_df = None
        self.forecast_series = None
        self.forecast_index = None


    def fit(self, seasonal=True, m=12):
        """
        Fits an ARIMA model to the training data for the target variable.
        Allows for seasonal adjustments if specified.
        Args:
        - seasonal (bool): Indicates if the model should consider seasonal effects.
        - m (int): The number of time steps for a single seasonal period.
        """
        # [TODO - API] auto_arima(self.training_data[self.target_variable], start_p=0, d=0, start_q=0, max_p=2, max_d=0, max_q ...
        self.model = auto_arima(self.training_data[self.target_variable], trace=True,
                                error_action='ignore', suppress_warnings=True,
                                seasonal=seasonal, m=m) ## stationary=True --> for data that is already stationary
        self.model.fit(self.training_data[self.target_variable])

        ## Preparing forecast
        forecast_periods = len(self.testing_data)
        self.forecast_df, self.conf_int = self.model.predict(n_periods=forecast_periods, return_conf_int=True)
        self.forecast_index = self.testing_data.index
        self.forecast_series = pd.Series(self.forecast_df, index=self.forecast_index)
        print("Model fitting and forecasting completed.")


    def validate(self):
        """
        Validates the model by plotting the forecast against actual values and calculates the Mean Absolute Error (MAE).
        It also visualizes confidence intervals if they are provided.
        """
        if self.forecast_series is None or self.testing_data is None:
            raise ValueError("Forecast or testing data is not available for validation.")

        ## DEBUG
        # sns.boxplot(x=self.training_data.ds, y=self.training_data.y)

        # Check if confidence intervals are available and correctly shaped
        conf_int_available = hasattr(self, 'conf_int') and self.conf_int.shape[0] == len(self.forecast_series)

        ## Calculate Mean Absolute Error (MAE)
        # mae = np.round(mean_absolute_error(self.testing_data[self.target_variable], self.forecast_series), 2)
        # mae = np.round(mean_absolute_error(self.testing_data[self.target_variable], self.forecast_series), 2)
        # print(f"Mean Absolute Error for {self.target_variable}: {mae}")

        plt.figure(figsize=(12, 6))
        plt.plot(self.training_data[self.target_variable], label='Train', color='blue')
        plt.plot(self.testing_data[self.target_variable], label='Test', color='green')
        plt.plot(self.forecast_series, label='Forecast', color='red')
        plt.fill_between(self.forecast_series.index, self.conf_int[:, 0], self.conf_int[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
        plt.title(f'Forecast vs Actuals for {self.target_variable} Prediction')
        plt.xlabel('Date')
        # plt.xticks(rotation=45)
        plt.ylabel(self.target_variable)
        plt.legend()
        # plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.show()


# In[11]:


## 'training_data1' and 'testing_data1' are for 'Penrose', and 'training_data2' and 'testing_data2' are for 'Takapuna'.
## Each dataset is a DataFrame containing datetime index and target variable column.
## Creating instances for Penrose and Takapuna for PM2.5 prediction
forecasting_penrose  = AdvancedTimeSeriesForecastingARIMA(training_data1, testing_data1, 'PM2.5')
forecasting_takapuna = AdvancedTimeSeriesForecastingARIMA(training_data2, testing_data2, 'PM2.5')

## Fitting models
print("🛠️ Fitting ARIMA models for Penrose ...")
forecasting_penrose.fit()
print("\n🛠️ Fitting ARIMA models for Takapuna ...")
forecasting_takapuna.fit()

## Validating models
print("🔬 Validating ARIMA models for Penrose ...")
forecasting_penrose.validate()
print("🔬 Validating ARIMA models for Takapuna ...")
forecasting_takapuna.validate()


# In[12]:


## 'training_data1' and 'testing_data1' are for 'Penrose', and 'training_data2' and 'testing_data2' are for 'Takapuna'.
## Each dataset is a DataFrame containing datetime index and target variable column.
## Creating instances for Penrose and Takapuna for PM2.5 prediction
forecasting_penrose  = AdvancedTimeSeriesForecastingARIMA(training_data1, testing_data1, 'PM10')
forecasting_takapuna = AdvancedTimeSeriesForecastingARIMA(training_data2, testing_data2, 'PM10')

## Fitting models
print("🛠️ Fitting ARIMA models for Penrose ...")
forecasting_penrose.fit()
print("\n🛠️ Fitting ARIMA models for Takapuna ...")
forecasting_takapuna.fit()

## Validating models
print("🔬 Validating ARIMA models for Penrose ...")
forecasting_penrose.validate()
print("🔬 Validating ARIMA models for Takapuna ...")
forecasting_takapuna.validate()


# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # ⚙️ Time-Series Analysis & Forecasting with Meta/Facebook's Prophet

# <div class="alert alert-block alert-info">
# 🎓 Time Series Forecasting with Meta/Facebook's Prophet
# 
# * Prophet works by decomposing the time series into three main components: trend, seasonality, and holidays. 
# * It uses an additive model where non-linear trends fit with yearly and weekly seasonalities, plus holidays. 
# * It accommodates seasonality with Fourier series and models holidays as indicator variables.
# </div>

# In[13]:


## Import Prophet class from fbprophet library
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot


# In[14]:


class AdvancedTimeSeriesForecastingProphet:
    
    def __init__(self, training_data, testing_data, target_variable, changepoint_prior_scale=0.01, interval_width=0.95, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, include_holidays=True, country_code='NZ'):
        """
        Initializes the class with training and testing data, target variable, and configurations for the Prophet model.
        Ensures data includes 'ds' and target_variable correctly named 'y'.

        Args:
            training_data (DataFrame): The training dataset with 'ds' and target_variable columns.
            testing_data (DataFrame): The testing dataset with 'ds' and target_variable columns.
            target_variable (str): The name of the column to forecast.
            changepoint_prior_scale (float): The flexibility of the automatic changepoint selection.
            daily_seasonality (bool): Whether to model daily seasonality.
            yearly_seasonality (bool): Whether to model yearly seasonality.
            interval_width (float): Uncertainty interval width. Set the uncertainty interval to 95% (the Prophet default is 80%)
            
            periods (int): Number of periods to forecast.
            freq (str): Frequency of the forecast (e.g., 'D' for daily).
            include_holidays (bool): If True, includes holiday effects based on the specified country.
            country_code (str): The ISO country code ('US') for including holidays.

        Returns:
            model (Prophet object): The fitted Prophet model.        
        """
        # self.validate_data(training_data, testing_data, target_variable)
        # self.training_data = training_data.rename(columns={target_variable: 'y', 'ds': 'ds'})
        # self.testing_data = testing_data.rename(columns={target_variable: 'y', 'ds': 'ds'})
        self.training_data = training_data.reset_index()[['ds', target_variable]].rename(columns={target_variable: 'y'}) ## target_variable --> 'y'
        self.testing_data = testing_data.reset_index()[['ds', target_variable]].rename(columns={target_variable: 'y'})
        self.target_variable = target_variable
        ## Initialize and configure the model in Prophet with the default parameters
        ## Set the uncertainty interval to 95% (the Prophet default is 80%)
        # self.model = Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.001)
        # self.model = Prophet(changepoint_prior_scale=0.004, seasonality_prior_scale=20, seasonality_mode='multiplicative', changepoint_range=0.9)
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale, 
            daily_seasonality=daily_seasonality, 
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            interval_width=interval_width)
        if include_holidays:
            self.model.add_country_holidays(country_name=country_code)  ## 'US' | 'NZ'
    
        self.forecast_df = None


    # def validate_data(self, training_data, testing_data, target_variable):
    #     """
    #     Validates the input data to ensure required columns are present.
    #     """
    #     required_columns = ['ds', target_variable]
    #     for data in [training_data, testing_data]:
    #         if not all(col in data.columns for col in required_columns):
    #             raise ValueError(f"Data must include the following columns: {required_columns}")

    ## Create a future dataframe for predictions, 24 hours a day | 168 hours a week into the future
    ## freq='MS' | 'D' | 'H'
    def fit(self, periods=7*24, freq='H'):
        """
        Fits the Prophet model using the training data.
        Generates predictions over the forecast horizon.

        Returns:
            forecast (DataFrame): The forecast results from Prophet.
            periods: We need to specify the number of days in future
        """
        ## Fit the model to the data
        self.model.fit(self.training_data)

        ## Making Future Predictions
        future = self.model.make_future_dataframe(periods=len(self.testing_data), freq=freq)
        ## Predicting the next quarter 120 days | 365 days (1 year) | period = 24 for daily seasonality; periods=24*7, freq='H')
        # future = self.model.make_future_dataframe(periods=periods, freq=freq)  
        
        self.forecast_df = self.model.predict(future)

        ## DEBUG
        ## Display the most critical output columns from the forecast
        # self.forecast_df[['ds','yhat','yhat_lower','yhat_upper']].tail()
        
        return self.forecast_df


    def validate(self):
        """
        Plots the forecasted values with uncertainty intervals alongside the actual testing data. 
        
        Validates the Forecasting Model by plotting the forecast with confidence intervals (highlights changepoints and the training data end) against actual values for a specified site. 
        Also shows components of the forecast, calculates MAE.
        """
        if self.forecast_df is None:
            raise ValueError("Forecast data is not available for validation.")

        ## DEBUG
        # sns.boxplot(x=self.training_data.ds, y=self.training_data.y)

        ## Extract the predicted and actual values for the visualization
        ## Ensure that forecasted data and actual data share the same index for proper comparison
        # forecast = self.forecast_df.set_index('ds')['yhat']
        forecast = self.forecast_df
        # actual = self.testing_data.set_index('ds')['y']
        actual_testing_data = self.testing_data
        last_training_date = self.training_data['ds'].max()

        ## Calculate Mean Absolute Error (MAE) only for overlapping dates
        # valid_indices = ~forecasted.isna()
        # mae = mean_absolute_error(actual[valid_indices], forecasted[valid_indices])
        # print(f"Mean Absolute Error (MAE): {mae:.2f}")

        ## [Plot 1] Plot the forecast: Shows the actual testing values, forecasted values from the fitted Model with the uncertainty intervals.
        # forecast_plot = self.model.plot(forecast, uncertainty=False, plot_cap=True)
        forecast_plot = self.model.plot(forecast, uncertainty=True)
        
        ## Adding changepoints and training end visual markers
        a = add_changepoints_to_plot(forecast_plot.gca(), self.model, forecast, cp_color='red')
        ## add a vertical line at the end of the training period
        axes = forecast_plot.gca()
        # last_training_date = pd.to_datetime('2021-12-31')
        axes.axvline(x=last_training_date, color='darkred', linestyle='solid', label='Training End') ## linestyle='--' | 'solid'
        
        ## plot true test data for the period after the red line
        actual_testing_data['ds'] = pd.to_datetime(actual_testing_data['ds'])
        plt.plot(actual_testing_data['ds'], actual_testing_data['y'],'ro', markersize=0.2, label='Actual Test Data')
        # actual = actual_testing_data.set_index('ds')['y']

        axes.set_title(f'[Prophet] Forecasting vs Actual Testing Data for {self.target_variable}')
        axes.set_xlabel('Date (ds)')
        axes.set_ylabel(f'{self.target_variable} (y)')
        axes.legend()
        axes.grid(True)
        plt.show()

        ## [Plot 2] Decomposes the forecast into trend, yearly seasonality, and weekly seasonality components.
        if hasattr(self.model, 'plot_components'):
            # fig2 = self.model.plot_components(self.forecast_df, uncertainty=False, plot_cap=True).show()
            fig2 = self.model.plot_components(self.forecast_df, uncertainty=True).show()
        
        ## Future Work:
        ## Change points: By default, Prophet automatically detects change points in the data. You can manually adjust their sensitivity or specify them.
        ## Seasonalities: Apart from built-in daily, weekly, and yearly seasonalities, you can add custom seasonalities (e.g., monthly).
        ## Holidays: You can add holidays and special events to improve the model’s accuracy on irregular occurrences.


# In[15]:


## DataFrame must have 'Site', 'ds', and other columns like 'PM2.5'
## Creating instances for Penrose and Takapuna for PM2.5 prediction
forecasting_prophet_penrose  = AdvancedTimeSeriesForecastingProphet(training_data1, testing_data1, 'PM2.5')
forecasting_prophet_takapuna = AdvancedTimeSeriesForecastingProphet(training_data2, testing_data2, 'PM2.5')

## Fitting models
print("🛠️ Fitting Prophet models for Penrose ...")
forecast = forecasting_prophet_penrose.fit()

print("🛠️ Fitting Prophet models for Takapuna ...")
forecasting_prophet_takapuna.fit()


## Validating models
print("🔬 Validating Prophet models for Penrose ...")
forecasting_prophet_penrose.validate()

print("🔬 Validating Prophet models for Takapuna ...")
forecasting_prophet_takapuna.validate()


# In[16]:


## DataFrame must have 'Site', 'ds', and other columns like 'PM2.5'
## Creating instances for Penrose and Takapuna for PM2.5 prediction
forecasting_prophet_penrose  = AdvancedTimeSeriesForecastingProphet(training_data1, testing_data1, 'PM10')
forecasting_prophet_takapuna = AdvancedTimeSeriesForecastingProphet(training_data2, testing_data2, 'PM10')

## DEBUG
# forecasting_prophet_takapuna.forecast_df[['ds','yhat','yhat_lower','yhat_upper']].head()

## Fitting models
print("🛠️ Fitting Prophet models for Penrose ...")
forecasting_prophet_penrose.fit()
print("🛠️ Fitting Prophet models for Takapuna ...")
forecasting_prophet_takapuna.fit()

## Validating models
print("🔬 Validating Prophet models for Penrose ...")
forecasting_prophet_penrose.validate()
print("🔬 Validating Prophet models for Takapuna ...")
forecasting_prophet_takapuna.validate()


# <div class="alert alert-block alert-info">
# TODO: 🎓 Time Series Forecasting with Facebook's Prophet - Executive Summary HERE ...
# </div>

# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # 🔧 Feature Selection

# <div class="alert alert-block alert-info">
# 🎓 In the section, ....</p>
# </div>

# In[17]:


def engineer_features(rawdata):
    """
    Adds time-based features to the dataframe to enhance the analysis capabilities, adjusting for the multi-level index.
    This includes extracting various temporal components and creating lag features for PM2.5 and PM10 variables to analyze time-based dependencies.
    """
    ## Extracting date-time components
    # timestamp_index = rawdata.index.get_level_values('Timestamp') ## 'Timestamp' <-- 'ds'
    timestamp_index = rawdata['Timestamp']
    # rawdata['Hour']       = timestamp_index.dt.hour
    rawdata['Day']        = timestamp_index.dt.day
    rawdata['DayOfWeek']  = timestamp_index.dt.dayofweek
    rawdata['Month']      = timestamp_index.dt.month
    rawdata['Quarter']    = timestamp_index.dt.quarter
    rawdata['Year']       = timestamp_index.dt.year

    # rawdata['DayOfYear']  = timestamp_index.dt.dayofyear
    ## Extract week of year for each timestamp
    rawdata['WeekOfYear'] = [d.isocalendar()[1] for d in timestamp_index]
    
    ## Calculating the season based on the month --> accurately reflects the local climate and seasonal cycles
    ## Season encoding: 1 (Summer): December, January, February; 2 (Autumn): March, April, May
    ##                  3 (Winter): June, July, August         ; 4 (Spring): September, October, November
    ## Adjusted for meteorological seasons in Auckland, New Zealand:  
    # rawdata['Season'] = (((timestamp_index.dt.month % 12) + 1) // 3) % 4 + 1 
    ## TODO: Southern Hemisphere like New Zealand and Australia vs Northern Hemisphere like England
    ## Correctly mapping the month to meteorological seasons for Southern Hemisphere (Auckland)
    rawdata['Season'] = rawdata['Month'].apply(
        lambda x: 1 if 9 <= x <= 11 else       ## Spring: Sep, Oct, Nov
                  2 if 12 <= x or x <= 2 else  ## Summer: Dec, Jan, Feb
                  3 if 3 <= x <= 5 else        ## Autumn: Mar, Apr, May
                  4                            ## Winter: Jun, Jul, Aug
    )

    ## Adding lag features for PM2.5 and PM10 to capture previous time steps' influence --> to compare the correlation with the other variables.
    rawdata['PM2.5_Lag1'] = rawdata.groupby('Site')['PM2.5'].shift(1)
    rawdata['PM2.5_Lag2'] = rawdata.groupby('Site')['PM2.5'].shift(2)
    rawdata['PM10_Lag1']  = rawdata.groupby('Site')['PM10'].shift(1)
    rawdata['PM10_Lag2']  = rawdata.groupby('Site')['PM10'].shift(2)
    ## Fill NaN with the next value in the column (backward fill)
    rawdata.fillna(method='bfill', inplace=True)
    ## Alternatively, fill NaN with the previous value in the column (forward fill)
    rawdata.fillna(method='ffill', inplace=True)
    
    if IS_DEBUG:
        rawdata[['Timestamp', 'Year', 'Quarter','Season', 'Month', 'Day', 'DayOfYear', 'DayOfWeek']].head()

    return rawdata


# In[19]:


## Apply the cleaning and querying function
# cleaned_data = rawdata.copy()
cleaned_data = engineer_features(rawdata.copy())

cleaned_data


# In[20]:


## 'Timestamp' as the only Ordinal Attribute/Column given its nature order in time-series data
ordinal_columns = ['Timestamp']

print("\n🎓 Describing the types of each attribute as numerical_columns (Continuous), ordinal_columns (Ordinal), or nominal_columns (Nominal) ...")
cleaned_numerical_columns, cleaned_nominal_columns = DescriptiveStatistics.describe_data(cleaned_data, ordinal_columns)

# targets = ['PM2.5', 'PM10', 'SO2']
# features = [feature for feature in df.columns if feature not in targets]


# In[22]:


cleaned_data1 = cleaned_data[cleaned_data['Site'] == 'Penrose']
cleaned_data2 = cleaned_data[cleaned_data['Site'] == 'Takapuna']

# if IS_DEBUG:
print("\n🎓 [Site1 - Penrose]  Summary Statistics of the {site1} cleaned_data1 Dataframe such as the mean, max/minimum values ...")
cleaned_data1.describe()    


# In[23]:


# if IS_DEBUG:
print("\n🎓 [Site2 - Takapuna]  Summary Statistics of the {site2} cleaned_data2 Dataframe such as the mean, max/minimum values ...")
cleaned_data2.describe()


# In[24]:


def get_top_correlated_features(data, target, num_features=12):
    """
    Calculates and returns the top N features most correlated with the target variable.
    
    Parameters:
    - data (DataFrame): The dataset containing numerical features.
    - target (str): The target variable for which correlations are calculated.
    - num_features (int): The number of top features to return.
    
    Returns:
    - List of top N correlated features.
    """
    ## Ensure data does not contain infinite or NaN values
    data = data.select_dtypes(include=[np.number]).dropna()
    
    ## Calculate the Pearson correlation matrix
    correlation_matrix = data.corr(method='pearson')
    
    ## Calculate absolute correlation with the target variable
    abs_corr_with_target = abs(correlation_matrix[target])
    
    ## Exclude the target variable and lag variables from the series
    features_to_exclude = [target] + [f"{target}_Lag1", f"{target}_Lag2"]
    abs_corr_with_target = abs_corr_with_target.drop(features_to_exclude, errors='ignore')
    
    ## Get the top N highly correlated features
    top_features = abs_corr_with_target.nlargest(num_features).index.tolist()
    
    return top_features


## [BACKUP]
# correlation_matrix = cleaned_data[cleaned_numerical_columns].corr(method='pearson')
# 
# ## Calculate absolute correlation with PM2.5
# abs_corr_with_pm25 = abs(correlation_matrix["PM2.5"])
# 
# ## Drop PM2.5, PM2.5_lag1 and PM2.5_lag2 from the series
# # abs_corr_with_pm25 = abs_corr_with_pm25.drop(['PM2.5', 'PM2.5_Lag1', 'PM2.5_Lag2'])
# 
# ## Get top 5 highly correlated features
# top_5_corr_features = abs_corr_with_pm25.nlargest(5).index.tolist()
# 
# print("Top 5 features highly correlated with PM2.5: ", top_5_corr_features)


# In[25]:


## Analyzing for both sites
top_features_data1 = get_top_correlated_features(cleaned_data1, 'PM2.5')
top_features_data2 = get_top_correlated_features(cleaned_data2, 'PM2.5')

print("Top 5 features highly correlated with PM2.5 in Penrose: ", top_features_data1)
print("Top 5 features highly correlated with PM2.5 in Takapuna: ", top_features_data2)


# In[26]:


## Analyzing for both sites
top_features_data1 = get_top_correlated_features(cleaned_data1, 'PM10')
top_features_data2 = get_top_correlated_features(cleaned_data2, 'PM10')

print("Top 5 features highly correlated with PM10 in Penrose: ", top_features_data1)
print("Top 5 features highly correlated with PM10 in Takapuna: ", top_features_data2)


# In[27]:


# import pandas as pd
# import matplotlib.pyplot as plt

# # Assuming rawdata and cleaned_data are pandas DataFrames loaded appropriately
# # Convert 'Timestamp' to datetime and set as index
# rawdata['Timestamp'] = pd.to_datetime(rawdata['Timestamp'])
# cleaned_data['Timestamp'] = pd.to_datetime(cleaned_data['Timestamp'])

# rawdata.set_index('Timestamp', inplace=True)
# cleaned_data.set_index('Timestamp', inplace=True)

def plot_pm_variation(data, feature, title, ax):
    """
    Helper function to plot PM2.5/PM10 variations on a given Axes object.
    
    Parameters:
    - data (DataFrame): The dataset to plot.
    - title (str): The title of the plot.
    - ax (Axes): The matplotlib Axes object where the plot will be drawn.
    """
    ax.plot(data[feature], label=f'{feature}')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{feature} Concentration')
    ax.set_title(title)
    ax.legend()

## BACKUP
# fig, axs = plt.subplots(1, 2, figsize=(20, 6))
# 
# # Plot the variation of PM2.5 in raw data
# axs[0].plot(rawdata['PM2.5'])
# axs[0].set_xlabel('Timestamp')
# axs[0].set_ylabel('PM2.5')
# axs[0].set_title('PM2.5 Variation over Time (Raw Data)')
# 
# # Plot the variation of PM2.5 in cleaned data
# axs[1].plot(cleaned_data['PM2.5'])
# axs[1].set_xlabel('Timestamp')
# axs[1].set_ylabel('PM2.5')
# axs[1].set_title('PM2.5 Variation over Time (Cleaned Data)')
# 
# plt.tight_layout()
# plt.show()


# In[28]:


fig, axs = plt.subplots(2, 2, figsize=(20, 12))  # Create a grid of 2x2 for Penrose and Takapuna

## Plotting
plot_pm_variation(rawdata_site1, 'PM2.5','Penrose - PM2.5 Variation over Time (Raw Data)', axs[0, 0])
plot_pm_variation(cleaned_data1, 'PM2.5','Penrose - PM2.5 Variation over Time (Cleaned Data)', axs[0, 1])
plot_pm_variation(rawdata_site2, 'PM2.5','Takapuna - PM2.5 Variation over Time (Raw Data)', axs[1, 0])
plot_pm_variation(cleaned_data2, 'PM2.5','Takapuna - PM2.5 Variation over Time (Cleaned Data)', axs[1, 1])

plt.tight_layout()
plt.show()


# In[29]:


# ## 'cleaned_data1' and 'cleaned_data2' are preloaded DataFrames for Penrose and Takapuna, respectively
# ## Convert 'Timestamp' to datetime and set as index
# cleaned_data1['Timestamp'] = pd.to_datetime(cleaned_data1['Timestamp'])
# cleaned_data2['Timestamp'] = pd.to_datetime(cleaned_data2['Timestamp'])
# 
# cleaned_data1.set_index('Timestamp', inplace=True)
# cleaned_data2.set_index('Timestamp', inplace=True)

## [DEBUG] Top 5 features - this should be dynamically determined from a correlation analysis
# top_features_data1 = ['AQI', 'PM10', 'NOx', 'Wind_Speed', 'Air_Temp']
# top_features_data2 = ['PM2.5', 'SO2', 'NO', 'Wind_Dir', 'Rel_Humidity']

def plot_features(data, features, site_name):
    fig, axs = plt.subplots(len(features), 1, figsize=(15, 15))

    for ax, feature in zip(axs, features):
        if feature in data.columns:
            ax.plot(data.index, data[feature], label=f'{feature} (Cleaned Data)')
            ax.set_title(f'Variation of {feature} in {site_name}')
            ax.set_xlabel('Date')
            ax.set_ylabel(feature)
            ax.grid(True)
        else:
            fig.delaxes(ax)  ## Remove the axis if the feature is not in the data

    plt.tight_layout()
    plt.show()


## [BACKUP]
# fig, axs = plt.subplots(3, 2, figsize=(15, 15))
# 
# # Flatten the axis array to make indexing easier
# axs = axs.flatten()
# 
# # Loop over each feature and its corresponding axis
# for ax, feature in zip(axs, top_5_corr_features):
#     ax.plot(cleaned_data[feature])
#     ax.set_title(f'Variation of {feature}')
#     ax.set_xlabel('Timestamp')
#     ax.set_ylabel(feature)
# 
# # If there are more subplots than features, delete the extra subplots
# if len(top_5_corr_features) < len(axs):
#     for ax in axs[len(top_5_corr_features):]:
#         fig.delaxes(ax)
# 
# plt.tight_layout()
# plt.show()


# In[30]:


## Call the function for each site
print("\n🎓 [Penrose] Visualizes the variation of the top N highly correlated features with PM2.5 across all times.\n")
plot_features(cleaned_data1, top_features_data1, 'Penrose')
print("\n🎓 [Takapuna] Visualizes the variation of the top N highly correlated features with PM2.5 across all times.\n")
plot_features(cleaned_data2, top_features_data2, 'Takapuna')


# In[31]:


def compute_summary_statistics(data, top_features):
    """
    Computes and prints summary statistics for selected features in a given dataset.
    
    Args:
    data (pd.DataFrame): The cleaned data for a specific site.
    top_features (list): List of features for which to compute statistics, determined from prior analysis.
    
    Returns:
    pd.DataFrame: Summary statistics of the selected features.
    """
    # Ensure all top features are in the data columns
    valid_features = [feature for feature in top_features if feature in data.columns]
    selected_features = ['PM2.5'] + valid_features

    # Get summary statistics
    summary_stats = data[selected_features].describe()
    return summary_stats


## [BACKUP]
# ## Select PM2.5 and top 5 features
# selected_features = ['PM2.5'] + top_5_corr_features
# 
# ## Get summary statistics
# summary_stats = cleaned_data[selected_features].describe()
# 
# ## Print summary statistics
# summary_stats


# In[32]:


## 'cleaned_data1' and 'cleaned_data2' are your cleaned datasets for Penrose and Takapuna respectively.
## 'top_features_data1' and 'top_features_data2' should be lists of top features determined from correlation analysis.

## [DEBUG] Check if the top features list is defined for each site
# top_features_data1 = ['NO2', 'Wind_Speed', 'Air_Temp', 'PM10', 'NOx']  # Example top features for Penrose
# top_features_data2 = ['SO2', 'NO', 'Rel_Humidity', 'Wind_Dir', 'NOx']  # Example top features for Takapuna

summary_stats_penrose = compute_summary_statistics(cleaned_data1, top_features_data1)
summary_stats_takapuna = compute_summary_statistics(cleaned_data2, top_features_data2)

print("🎓 Summary Statistics for Penrose:")
summary_stats_penrose


# In[33]:


print("\n🎓 Summary Statistics for Takapuna:")
summary_stats_takapuna


# <div class="alert alert-block alert-info">
# 🎓 Interpretation ....</p>
# 
# * **Target**:
# 
# * **Features**:
#     * **NO2**
# 
# </div>

# # References

# [Forecasting: Principles and Practice (3rd ed)](https://otexts.com/fpp3) : Rob J Hyndman and George Athanasopoulos; Monash University, Australia 

# <footer style="padding-bottom:35px; background:#f9f9f9; border-bottom:3px solid #00b2b1">
#     <div style="float:left;margin-top:14px;color:#E37C4D">Predicting Air Particulate Matter at Scale ⛅️</div>
#     <div style="float:right;">
#         <div style="float:left; margin-top:14px">
#             Auckland University of Technology (AUT) 🎓
#         </div>
#     </div>
# </footer>
