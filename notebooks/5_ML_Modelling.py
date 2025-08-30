#!/usr/bin/env python
# coding: utf-8

# <header style="padding:1px;background:#f9f9f9;border-top:3px solid #00b2b1"><img id="Teradata-logo" src="https://www.teradata.com/Teradata/Images/Rebrand/Teradata_logo-two_color.png" alt="Teradata" width="220" align="right" />
# 
# <b style = 'font-size:28px;font-family:Arial;color:#E37C4D'>🎓 Predicting Air Particulate Matter at Scale ⛅️</br> 🛠️ 5. Machine Learning Modelling</b>
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
# 🎓 This notebook shows various methods that can be used to predict a Environmental Dataset using Machine Learning. It should be ready for reuse in the next steps (Intelligent Dashboard) in CRISP-DM for Data Science.
#     
# </div>

# * **Workflow steps:**
# 
#   1. Import the required teradataml modules.
#      * Connect to a Vantage system.
#   2. ⚙️ Machine Learning Modelling
#   3. Cleanup.

# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
#     
# # 🎯 Libraries and Reusable Functions 

# <div class="alert alert-block alert-info">
# 🎓 This section executes all of the cells in <b>DataFrameAdapter.ipynb</b>.
# </div>

# In[1]:


import os, logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## .env --> Setting the environment variable for output format programmatically
os.environ['ENV_PATH'] = 'cleaned_vantage.env'  ## Set to `.env_cleaned` or `.env_raw` as needed

# %run -i ./Data_Loading_and_Descriptive_Statistics.ipynb
get_ipython().run_line_magic('run', '-i ./DataFrameAdapter.ipynb')


# In[2]:


# cleaned_data_site1
cleaned_data_site2


# In[3]:


# def ensure_numeric(data):
#     """
#     Ensure all columns in the DataFrame are numeric. Convert non-numeric columns to numeric where possible.

#     Args:
#     data (pd.DataFrame): The DataFrame to be processed.

#     Returns:
#     pd.DataFrame: DataFrame with all columns converted to numeric.
#     """
#     if 'Site' in data.columns:
#         data.drop(columns=['Site'], inplace=True)
#     for col in data.columns:
#         if data[col].dtype == 'object':
#             data[col] = pd.to_numeric(data[col], errors='coerce')
#     return data

# ## Convert non-numeric columns to numeric
# cleaned_data_site1 = ensure_numeric(cleaned_data_site1)
# cleaned_data_site2 = ensure_numeric(cleaned_data_site2)

# logging.info("Non-numeric columns converted to numeric where possible.")


# In[4]:


# ## Extract and select features for both sites and multiple pollutants

# ## 'PM2.5' Analyzing for both sites
# top_features_data11 = DataFrameAdapter.extract_featuretools_features(data=cleaned_data_site1, target_column='PM2.5', entity_id='site1')
# top_features_data12 = DataFrameAdapter.extract_featuretools_features(data=cleaned_data_site2, target_column='PM2.5', entity_id='site2')

# logging.info("\n🌟 top_features_data11: Top Featuretools features highly correlated with PM2.5 in Penrose: %s\n", top_features_data11.head())
# logging.info("\n🌟 top_features_data12: Top Featuretools features highly correlated with PM2.5 in Takapuna: %s\n", top_features_data12.head())

# ## 'PM10' Analyzing for both sites
# top_features_data21 = DataFrameAdapter.extract_featuretools_features(data=cleaned_data_site1, target_column='PM10', entity_id='site1')
# top_features_data22 = DataFrameAdapter.extract_featuretools_features(data=cleaned_data_site2, target_column='PM10', entity_id='site2')

# logging.info("\n🌟 top_features_data21: Top Featuretools features highly correlated with PM10 in Penrose: %s\n", top_features_data21.head())
# logging.info("\n🌟 top_features_data22: Top Featuretools features highly correlated with PM10 in Takapuna: %s\n", top_features_data22.head())


# In[5]:


# ## Extract and select features for both sites and multiple pollutants

# ## 'PM2.5' Analyzing for both sites
# top_features_data11 = DataFrameAdapter.extract_tsfresh_features(data=cleaned_data_site1, target_column='PM2.5')
# top_features_data12 = DataFrameAdapter.extract_tsfresh_features(cleaned_data_site2, target_column='PM2.5')

# logging.info("\n🌟 top_features_data11: Top Tsfresh features highly correlated with PM2.5 in Penrose: %s\n", top_features_data11)
# logging.info("\n🌟 top_features_data12: Top Tsfresh features highly correlated with PM2.5 in Takapuna: %s\n", top_features_data12)

# ## 'PM10' Analyzing for both sites
# top_features_data21 = DataFrameAdapter.extract_tsfresh_features(data=cleaned_data_site1, target_column='PM10')
# top_features_data22 = DataFrameAdapter.extract_tsfresh_features(cleaned_data_site2, target_column='PM10')

# logging.info("\n🌟 top_features_data21: Top Tsfresh features highly correlated with PM10 in Penrose: %s\n", top_features_data21)
# logging.info("\n🌟 top_features_data22: Top Tsfresh features highly correlated with PM10 in Takapuna: %s\n", top_features_data22)


# In[6]:


## 'PM2.5' Analyzing for both sites
top_features_data11 = DataFrameAdapter.get_top_correlated_features(data=cleaned_data_site1, target='PM2.5', num_features=10)
top_features_data12 = DataFrameAdapter.get_top_correlated_features(data=cleaned_data_site2, target='PM2.5', num_features=10)

logging.info("\n🌟 top_features_data11: Top features for PM2.5 in Penrose: %s\n", top_features_data11)
logging.info("\n🌟 top_features_data12: Top features for PM2.5 in Takapuna: %s\n", top_features_data12)

## 'PM10' Analyzing for both sites
top_features_data21 = DataFrameAdapter.get_top_correlated_features(data=cleaned_data_site1, target='PM10', num_features=10)
top_features_data22 = DataFrameAdapter.get_top_correlated_features(data=cleaned_data_site2, target='PM10', num_features=10)

logging.info("\n🌟 top_features_data21: Top features for PM10 in Penrose: %s\n", top_features_data21)
logging.info("\n🌟 top_features_data22: Top features for PM10 in Takapuna: %s\n", top_features_data22)


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 🛠️ [Predictive Analytics] Predictive Models

# <header style="padding:1px;border-top:3px solid green"></header>
# 
# 
# <details>
#     <summary>Adaptive Cross Validation: TimeSeriesSplit with Time Series Data vs. KFold/Standard Cross Validation</summary>
# 
# </details>
# 
# | TimeSeriesSplit Adaptive Cross Validation | KFold/Standard Cross Validation |
# |---|---|
# | ![TimeSeriesSplit Cross Validation](https://raw.githubusercontent.com/nnthanh101/Analytics-Experience/main/docs/static/img/data-science/Cross-Validation-Time-Series.png?token=GHSAT0AAAAAACOP57OLLKHNK4B7PAJ7FYJEZSUHFHQ) | ![KFold/Standard Cross Validation](https://raw.githubusercontent.com/nnthanh101/Analytics-Experience/main/docs/static/img/data-science/Cross-Validation.png?token=GHSAT0AAAAAACOP57OKBO2CBT4BJXYZ2ZY4ZSUHFXA) |
# 

# In[63]:


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
# from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from prophet import Prophet
from xgboost import XGBRegressor

import joblib, pickle


# In[9]:


# !pip install neuralprophet
from neuralprophet import NeuralProphet


# In[10]:


class PredictiveModels:
    """
    A class for creating and managing various predictive models.
    
    * [x] The adaptive forecasting approach to enhance long-term predictive accuracy for RQ2: 
      * Use TimeSeriesSplit for cross-validation.
      * Continuously update the model with new data during each n_splits.
    * 
    """
    # def __init__(self, training_data, testing_data, target_variable, features):
    def __init__(self, training_data, target_variable, features):
        """
        * [x] Initialize the predictor with training data.
        * [x] Removed test_data, since test data is derived correctly during cross-validation.

        :param training_data: DataFrame containing the training data.
        :param testing_data: DataFrame containing the testing data.
        :param target_variable: The target variable for prediction (e.g., 'PM2.5' or 'PM10').
        :param features: List of feature columns to be used for training.
        """
        ## Store training data with only the relevant features and target variable; also refer to adaptive_cross_validate()
        # self.training_data = training_data
        self.training_data = training_data[features + [target_variable]]
        # self.testing_data    = testing_data
        self.target_variable = target_variable
        self.features = features
        
        ## Extract features and target from training and testing data
        self.X_train = self.training_data[self.features]
        self.y_train = self.training_data[self.target_variable]
        # self.X_test = self.testing_data[self.features]
        # self.y_test = self.testing_data[self.target_variable]

        ## Standardize the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        # self.X_test_scaled  = self.scaler.transform(self.X_test)

        ## Initialize dictionaries to store models and evaluation results
        self.models = {}              ## Dictionary to store models
        self.evaluation_results = {}  ## Dictionary to store evaluation results
        logging.debug(f"Initialized PredictiveModels with target: {target_variable}")

#### Step 2: Model Creation

    def _create_model(self, model_name, arima_model=None):
        """
        Create and configure a model instance based on the model name.

        :param model_name: Name of the model to be created.
        :param arima_model: ARIMA model order string if applicable (e.g., 'ARIMA(1,1,1)').
        
        :return: Configured model instance.
        """
        if model_name == 'LinearRegression':
            return LinearRegression()
        elif model_name == 'Ridge':
            ## Consider using 'Ridge', 'Lasso' regression for regularization to prevent overfitting
            return Ridge(alpha=1.0)  ## Adjust alpha for regularization strength
        elif model_name == 'Lasso':
            return Lasso(alpha=0.1)  ## Regularization strength
        elif model_name == 'RandomForest':
            # return RandomForestRegressor(n_estimators=100, random_state=42)
            return RandomForestRegressor(
                n_estimators=200,      ## Increase number of trees
                max_depth=10,          ## Limit depth of trees
                min_samples_split=5,   ## Minimum number of samples required to split a node
                min_samples_leaf=2,    ## Minimum number of samples required at each leaf node
                random_state=42,
                bootstrap=True
            )
        elif model_name == 'SVR':
            ## Support Vector Regression (SVR) using linear kernel vs non-linear Radial Basis Function (RBF) kernel: 'linear' | 'poly' | 'rbf'
            # return SVR(kernel='rbf')
            # if kernel == 'linear':
            #     return SVR(kernel='linear', C=1.0)
            # elif kernel == 'poly':
            #     return SVR(kernel='poly', C=1.0, degree=3)
            return SVR(kernel='rbf', C=1.0, gamma='scale') ## Experiment with different kernels and hyperparameters like C and gamma
        elif model_name == 'XGBoost':
            # return XGBRegressor(objective='reg:squarederror', n_estimators=100)
            return XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,    ## Increase number of trees
                learning_rate=0.05,  ## Lower learning rate
                max_depth=6,         ## Limit depth of trees
                subsample=0.8,       ## Subsample ratio of training instances
                colsample_bytree=0.8,  ## Subsample ratio of columns
                random_state=42
            )
        elif model_name == 'ARIMA' and arima_model is not None:
            try:
                order = tuple(map(int, arima_model.split('(')[1].strip(')').split(',')))
                model = pm.ARIMA(order=order, suppress_warnings=True)
                logging.debug(f"ARIMA model created with order: {order}")
                return model
            except Exception as e:
                logging.error(f"Error in creating ARIMA model: {e}")
                raise ValueError(f"Error in creating ARIMA model: {e}")
        elif model_name == 'Prophet':
            return self._create_prophet_model()
        elif model_name == 'NeuralProphet':
            return self._create_neural_prophet_model()
        elif model_name == 'LSTM':
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(1, len(self.features))))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            return model
        elif model_name == 'MLP':
            model = Sequential()
            model.add(Dense(100, activation='relu', input_dim=len(self.features)))
            model.add(Dense(50, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            return model
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

    def _create_prophet_model(changepoint_prior_scale=0.01, interval_width=0.95, 
                            daily_seasonality=True, weekly_seasonality=True, 
                            yearly_seasonality=True, include_holidays=False, country_code=None):
        """
        Configures and returns a Prophet model with the specified parameters.
    
        :param changepoint_prior_scale: Flexibility of the trend
        :param interval_width: Uncertainty interval width
        :param daily_seasonality: Whether to include daily seasonality
        :param weekly_seasonality: Whether to include weekly seasonality
        :param yearly_seasonality: Whether to include yearly seasonality
        :param include_holidays: Whether to include country-specific holidays
        :param country_code: Country code for holidays
    
        :return: Configured Prophet model
        """
        # logging.info(f"Configuring Prophet model with changepoint_prior_scale={changepoint_prior_scale}, interval_width={interval_width}, "
        #              f"daily_seasonality={daily_seasonality}, weekly_seasonality={weekly_seasonality}, yearly_seasonality={yearly_seasonality}, "
        #              f"include_holidays={include_holidays}, country_code={country_code}")
        model = Prophet(
                        # changepoint_prior_scale=changepoint_prior_scale, 
                        # interval_width=interval_width, 
                        daily_seasonality=daily_seasonality, 
                        weekly_seasonality=weekly_seasonality, 
                        yearly_seasonality=yearly_seasonality)
        
        if include_holidays and country_code:
            model.add_country_holidays(country_name=country_code)

        logging.debug("Prophet model configured successfully")
        return model

    def _create_neural_prophet_model(self, periods=7*24, n_forecasts=1):
        """
        Fits the NeuralProphet model using the training data and generates future predictions.
        
        :param df: DataFrame containing the training data with columns 'ds' and 'y'.
        :param periods: Number of periods for future predictions.
        :param n_forecasts: Number of steps ahead to forecast.
        :return: The forecast results from NeuralProphet.
        """
        model = NeuralProphet(
            n_changepoints=0,              ## Disable trend changepoints
            yearly_seasonality=False,      ## Disable yearly seasonality
            weekly_seasonality=True,       ## Enable weekly seasonality
            daily_seasonality=True,        ## Enable daily seasonality
            n_lags=24,                     ## Use 24 lags (one day of hourly data)
            n_forecasts=n_forecasts        ## Forecast 1 step ahead
        )
        
        logging.info(f"NeuralProphet model generated for {periods} periods.")
        return model
    
#### Step 3: Fitting the Model
    
    def fit(self, model_name, model=None, param_grid=None):
        """
        Fit a model to the training data.

        The fit() method is used within the adaptive_cross_validate() function to adapt the model with new test data included in the training. 
        This method needs to ensure that the model is continuously updated with new data in each fold of the cross-validation.

        :param model_name: Name of the model (e.g., 'LinearRegression', 'RandomForest', 'SVR', 'LSTM', 'MLP').
        :param model: The instantiated model to be trained (optional for custom models).
        """
        if model is None:
            model = self._create_model(model_name)

        logging.debug(f"Fitting Model: {model_name}")
        if model_name in ['LSTM', 'MLP']:
            self._fit_deep_learning_model(model_name, model)
        elif model_name == 'Prophet':
            df_train = pd.DataFrame({'ds': self.training_data.index[self.X_train], 'y': self.y_train})
            df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')
            df_train = df_train.dropna()
            if len(df_train) > 0:  ## Ensure non-empty DataFrame
                model.fit(df_train)
        elif model_name == 'NeuralProphet':
            df_train = pd.DataFrame({'ds': self.training_data.index, 'y': self.y_train})
            # df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')
            model.fit(df_train)
        elif model_name == 'ARIMA':
            model.fit(self.y_train)
        else:
            ## Standard fitting process for other models
            # if param_grid:
            #     grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
            #     grid_search.fit(self.X_train_scaled, self.y_train)
            #     model = grid_search.best_estimator_
            # else:
            model.fit(self.X_train_scaled, self.y_train)  ## Fit traditional ML models

        self.models[model_name] = model   ## Store the model in the models dictionary
        ## Fit the model only once per fold and save only after cross-validation is complete: f"data/models/{model_name}.pkl":  save_format='joblib' | 'pickle'
        # self.save_model(model_name, model, save_format='joblib')
        logging.info(f'Model {model_name} fitted successfully')

    # def prophet_fit(self, periods=7*24, freq='H'):
    #     """
    #     Fits the Prophet model using the training data and generates predictions over the forecast horizon.
    #     :param periods: Number of periods for future predictions.
    #     :param freq: Frequency of the data ('H' for hourly).
    #     :return: The forecast results from Prophet.
    #     """
    #     self.model.fit(self.training_data)
    #     future = self.model.make_future_dataframe(periods=periods, freq=freq)
    #     self.forecast_df = self.model.predict(future)
    #     logging.info(f"Prophet model fit and forecast generated for {periods} periods with frequency {freq}.")
    #     return self.forecast_df

    def _fit_deep_learning_model(self, model_name, model):
        """
        Fit a deep learning model (LSTM and MLP).

        :param model_name: Name of the model (e.g., 'LSTM', 'MLP').
        :param model: The instantiated deep learning model to be trained.
        """
        if model_name == 'LSTM':  ## Reshape data for LSTM
            X_train_reshaped = self.X_train_scaled.reshape((self.X_train_scaled.shape[0], 1, self.X_train_scaled.shape[1]))
            model.fit(X_train_reshaped, self.y_train, epochs=50, batch_size=72, verbose=2, shuffle=False)
        else:  ## Fit MLP model
            model.fit(self.X_train_scaled, self.y_train, epochs=50, batch_size=10, verbose=2)


    # def predict(self, model_name):
    #     """
    #     Make predictions using a trained model.

    #     :param model_name: Name of the model to be used for prediction.
    #     :return: Array of predictions.
    #     """
    #     model = self.models.get(model_name)
    #     if not model:
    #         # raise ValueError(f"Model '{model_name}' has not been trained.")
    #         # model = joblib.load(f"data/models/{model_name}.pkl")  ## Load the model from disk if not found in memory
    #         model = self.load_model(model_name)  ## Load the model from disk

    #     if model_name == 'LSTM':
    #         X_test_reshaped = self.X_test_scaled.reshape((self.X_test_scaled.shape[0], 1, self.X_test_scaled.shape[1]))
    #         return model.predict(X_test_reshaped).flatten()
    #     return model.predict(self.X_test_scaled)


    def evaluate(self, model_name):
        """
        Evaluate the model using several metrics.

        :param model_name: Name of the model to be evaluated.
        :return: Dictionary of evaluation metrics.
        """
        ## Predictions using the specified model
        # predictions = self.predict(model_name)
        # predictions = self.all_predictions if hasattr(self, 'all_predictions') else self.predict(model_name)
        # y_test = self.all_y_test if hasattr(self, 'all_y_test') else self.y_train
        if hasattr(self, 'all_predictions') and hasattr(self, 'all_y_test'):
            predictions = self.all_predictions
            y_test = self.all_y_test
        else:
            predictions = self.predict(model_name)
            y_test = self.y_test

        
        ## Ensure the length of predictions and y_test are the same
        if len(predictions) != len(y_test):
            raise ValueError("Inconsistent number of samples between predictions and true values")
        
        ## Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, predictions)
        ## Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        ## Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, predictions)
        ## Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        ## Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
        # smape = np.mean(2 * np.abs(y_test - predictions) / (np.abs(y_test) + np.abs(predictions))) * 100
        # ## Calculate Median Absolute Percentage Error (MDAPE)
        # mdape = np.median(np.abs((self.y_test - predictions) / self.y_test)) * 100
        # ## Calculate Geometric Mean Relative Absolute Error (GMRAE)
        # gmrae = np.exp(np.mean(np.log(np.abs((self.y_test - predictions) / self.y_test)))) * 100

        ## Calculate R-squared (R2)
        r2 = r2_score(y_test, predictions)
        ## Calculate Adjusted R-squared (Adjusted R2)
        adj_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-self.X_test_scaled.shape[1]-1)

        ## Store evaluation metrics in a dictionary
        self.evaluation_results[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            # 'SMAPE': smape,
            # 'MDAPE': mdape,
            # 'GMRAE': gmrae,
            'R2': r2,
            'Adjusted R2': adj_r2
        }

        ## Log results sorted by the preferred metric (RMSE)
        results = sorted(self.evaluation_results.items(), key=lambda x: x[1]['RMSE'])
        for model, metrics in results:
            logging.info(
                f"Model: {model_name}\n"
                f"RMSE: {rmse:.2f}, MSE: {mse:.2f}, "
                f"MAE: {mae:.2f}, MAPE: {mape:.2f}%, "
                # f"SMAPE: {smape:.2f}%, MDAPE: {mdape:.2f}%, GMRAE: {gmrae:.2f}%\n"
                f"R2: {r2:.2f}, Adjusted R2: {adj_r2:.2f}"
            )

        best_model = results[0][0]
        logging.debug(f"\nThe best RMSE from evaluation_results: {best_model}")
        
        return self.evaluation_results[model_name]

#### Step 4: Cross-Validation

    def adaptive_cross_validate(self, model_name, arima_model=None, site='Penrose', pollutant='PM2.5', n_splits=5, n_forecasts=24):
        """
        [x] Perform adaptive cross-validation using TimeSeriesSplit.
        [ ] Perform K-Fold Cross-Validation.

        :param model_name: Name of the model to be cross-validated.
        :param n_splits: Number of splits in TimeSeriesSplit (or Number of cross-validation folds).
        :param n_forecasts: Number of steps ahead to forecast for NeuralProphet. Multi-step Forecasting for 24 hours ahead.
        :return: List of cross-validation scores.
        """
        # kf = KFold(n_splits=n_splits, shuffle=False, random_state=42)
        ts_cross_validate = TimeSeriesSplit(n_splits=n_splits)
        # X = self.training_data[self.features].values
        X = self.training_data[self.features]         ## Retain DataFrame with column names
        # y = self.training_data[self.target_variable].values
        y = self.training_data[self.target_variable]  ## Retain DataFrame with column names
        n_predictors = X.shape[1]  ## Number of predictors
        
        # scores = {'RMSE': [], 'MSE': [], 'MAE': [], 'MAPE': [], 'R2': [], 'Adjusted R2': []}
        all_metrics = {
            'RMSE': [], 'MSE': [], 'MAE': [], 'MAPE': [], 'R2': [], 'Adjusted R2': []
        }
        all_predictions = []
        all_y_test = []

        ## Create the Prophet model once
        # if model_name == 'Prophet':
        #     prophet_model = self._create_prophet_model(include_holidays=True, country_code='NZ')

        ## Split & print out results
        for fold, (train_index, test_index) in enumerate(ts_cross_validate.split(X)):
            logging.info(f"Processing fold {fold + 1}/{n_splits} for {model_name}")
    
            ## Splitting data into training and testing sets for current fold
            # X_train_ts, X_test_ts = X[train_index], X[test_index]
            # y_train_ts, y_test_ts = y[train_index], y[test_index]
            X_train_ts, X_test_ts = X.iloc[train_index], X.iloc[test_index]  ## Retain DataFrame with column names
            y_train_ts, y_test_ts = y.iloc[train_index], y.iloc[test_index]  ## Retain DataFrame with column names

            ## Scaling the data
            X_train_ts_scaled = self.scaler.fit_transform(X_train_ts)
            X_test_ts_scaled = self.scaler.transform(X_test_ts)

            model = self._create_model(model_name, arima_model=arima_model)

            if model_name in ['LSTM', 'MLP']:
                model = self._reinitialize_model(model_name) ## Reinitialize the model for each split/fold starts with an untrained model
                if model_name == 'LSTM':
                    X_train_ts_reshaped = X_train_ts_scaled.reshape((X_train_ts_scaled.shape[0], 1, X_train_ts_scaled.shape[1]))
                    X_test_ts_reshaped = X_test_ts_scaled.reshape((X_test_ts_scaled.shape[0], 1, X_test_ts_scaled.shape[1]))
                    model.fit(X_train_ts_reshaped, y_train_ts, epochs=50, batch_size=72, verbose=0, shuffle=False)
                    predictions_ts = model.predict(X_test_ts_reshaped).flatten()
                else:
                    model.fit(X_train_ts_scaled, y_train_ts, epochs=50, batch_size=10, verbose=0)
                    predictions_ts = model.predict(X_test_ts_scaled).flatten()
            elif model_name == 'ARIMA':
                # logging.info(f"Training fold {fold + 1} for model {model_name}")
                model.fit(y_train_ts)  # Fitting the model with training data for the current fold
                # logging.debug(f"Completed training fold {fold + 1} for model {model_name}")
                predictions_ts = model.predict(n_periods=len(y_test_ts))  ## Use predict method of pmdarima
            elif model_name == 'Prophet':
                try:
                    ## Create Prophet model once and reuse if possible
                    # if 'prophet_model' not in self.models:
                    #     self.models['prophet_model'] = self._create_prophet_model(include_holidays=True, country_code='NZ')
                    # model = self.models['prophet_model']
                    model = self._create_prophet_model(include_holidays=True, country_code='NZ')

                    ## Prepare training data for Prophet
                    df_train = pd.DataFrame({'ds': self.training_data.index[train_index], 'y': y_train_ts})
                    df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')
                    
                    ## Fit the model
                    model.fit(df_train)
                    logging.debug(f"Prophet model fitted for fold {fold + 1}")

                    ## Prepare test data for Prophet
                    df_test = pd.DataFrame({'ds': self.training_data.index[test_index]})
                    df_test['ds'] = pd.to_datetime(df_test['ds'], errors='coerce')  ## Handle invalid dates

                    ## Predict using the fitted model
                    predictions_ts = model.predict(df_test)['yhat'].values          ## prediction method
                    logging.debug(f"Predictions made for fold {fold + 1} using Prophet model")
                    
                except Exception as e:
                    logging.error(f"Error in fitting/predicting with Prophet model for fold {fold + 1}: {e}")
                    predictions_ts = np.zeros(len(test_index))

                self.models[model_name] = model  ## Ensure model is added to the dictionary
                logging.debug(f"Completed training fold {fold + 1} for model {model_name}")

            elif model_name == 'NeuralProphet':
                model = self._create_neural_prophet_model(periods=len(test_index), n_forecasts=n_forecasts)
                ## Prepare training data for NeuralProphet
                df_train = pd.DataFrame({'ds': self.training_data.index[train_index], 'y': y_train_ts})
                df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')
                ## Fit the model
                model.fit(df_train)
                ## Prepare test data for NeuralProphet
                df_test = pd.DataFrame({'ds': self.training_data.index[test_index], 'y': y_test_ts})  ## Add 'y' column to df_test
                df_test['ds'] = pd.to_datetime(df_test['ds'], errors='coerce')
                ## Predict using the fitted model
                # future = model.make_future_dataframe(df_test, n_historic_predictions=True, periods=len(test_index), freq='D')
                future = model.make_future_dataframe(df_test, n_historic_predictions=True, periods=len(test_index))
                # predictions_ts = model.predict(future)['yhat1'].values: yhat, yhat1 ...
                predictions_ts = model.predict(future)[f'yhat{n_forecasts}'].values
                if np.isnan(predictions_ts).any():
                    logging.warning(f"NaN values found in predictions for fold {fold + 1}. Replacing NaNs with mean value.")
                    predictions_ts = np.nan_to_num(predictions_ts, nan=np.nanmean(predictions_ts))
                ## Ensure the lengths of y_test_ts and predictions_ts are consistent
                predictions_ts = predictions_ts[:len(y_test_ts)]
                self.models[model_name] = model  ## Store the model in the models dictionary
                logging.debug(f"Predictions made for fold {fold + 1} using NeuralProphet model")

            else:
                # if param_grid:
                #     grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
                #     grid_search.fit(X_train_ts_scaled, y_train_ts)
                #     model = grid_search.best_estimator_
                # else:
                logging.debug(f"Training fold {fold + 1} for model {model_name}")
                model.fit(X_train_ts_scaled, y_train_ts)
                logging.debug(f"Completed training fold {fold + 1} for model {model_name}")
                predictions_ts = model.predict(X_test_ts_scaled)

            ## Evaluate the model
            # scores.append(r2_score(y_test_ts, predictions_ts))
            y_test_ts = y_test_ts[:len(predictions_ts)]
            # metrics = CommonUtils.calculate_metrics(y_test_ts, predictions_ts)
            metrics = CommonUtils.calculate_metrics_adj_r2(y_test_ts, predictions_ts, n_predictors)
            for metric_name, metric_value in metrics.items():
                all_metrics[metric_name].append(metric_value)
            logging.debug(f"Fold results for {model_name}: all_metrics = {all_metrics}")
            
            all_predictions.extend(predictions_ts)
            all_y_test.extend(y_test_ts)

            ## Adapt the model with the new test data included in training
            self.X_train_scaled = np.vstack([self.X_train_scaled, X_test_ts_scaled])
            self.y_train = np.concatenate([self.y_train, y_test_ts])
            self.X_test_scaled = X_test_ts_scaled
            self.y_test = y_test_ts
            
            # self.fit(model_name, model)  ## Call fit method with the current fold's model and data
            if model_name != 'Prophet' and model_name != 'NeuralProphet':  ## Avoid redundant fitting for Prophet
                self.fit(model_name, model)

        self.all_predictions = np.array(all_predictions)
        self.all_y_test = np.array(all_y_test)

        logging.info(f"Adaptive cross-validation scores for {model_name}: {all_metrics}")
        ## Save the model after all folds are completed
        logging.debug(f"Saving model {model_name} after all folds")
        self.save_model(model_name, model, save_format='joblib',site=site, pollutant=pollutant)
        
        # return scores
        return all_metrics


    def _reinitialize_model(self, model_name):
        """
        Reinitialize a deep learning model.

        :param model_name: Name of the model to be reinitialized.
        :return: A new instance of the deep learning model.
        """
        if model_name == 'LSTM':
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(1, self.X_train.shape[1])))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            return model
        elif model_name == 'MLP':
            model = Sequential()
            model.add(Dense(100, activation='relu', input_dim=self.X_train.shape[1]))
            model.add(Dense(50, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            return model

    def model_selection(self, models, preference='RMSE'):
        """
        Compare multiple models based on configurable preference metrics including RMSE, MAE, MAPE, AIC, and BIC.

        :param models: List of tuples. Each tuple contains the model name and instantiated model.
        :param preference: Preferred metric for model evaluation.
        :return: Best model based on the preferred metric.
        """
        results = []
        best_model = None
        best_metric = float('inf')

        for name, model in models:
            self.fit(name, model)
            evaluation = self.evaluate(name)
            results.append((name, evaluation))

            if evaluation[preference] < best_metric:
                best_metric = evaluation[preference]
                best_model = name

        return best_model, results


    # def hyperparameter_tuning(self, model_name, param_grid):
    #     model = self._create_model(model_name)
    #     tscv = TimeSeriesSplit(n_splits=5)
    #     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error')
    #     grid_search.fit(self.X_train_scaled, self.y_train)
    #     best_params = grid_search.best_params_
    #     logging.info(f"Best parameters for {model_name}: {best_params}")
    #     self.fit(model_name, best_params)
    #     return best_params

#### Step 5: Save and Load Models
    
    def save_model(self, model_name, model, save_format='joblib', site='Penrose', pollutant='PM2.5'):
        """
        Save the fitted model to disk.
    
        :param model_name: Name of the model.
        :param model: The trained model instance.
        :param save_format: Format to save the model ('joblib' or 'pickle').
        :param site: Site information (e.g., 'Penrose' or 'Takapuna').
        :param pollutant: Pollutant information (e.g., 'PM2.5' or 'PM10').
        """
        logging.info(f'Saving model {model_name} for site {site} and pollutant {pollutant}')
        file_extension = 'pkl' if save_format == 'pickle' else 'joblib'
        file_path = f"data/models/{site}_{pollutant}_{model_name}.{file_extension}"
        
        if save_format == 'joblib':
            joblib.dump(model, file_path)
        elif save_format == 'pickle':
            with open(file_path, 'wb') as file:
                pickle.dump(model, file)
        elif save_format == 'pmml':
            from nyoka import skl_to_pmml  ## for sklearn models
            skl_to_pmml(model, file_path)  ## This line depends on the specific implementation and libraries used
        else:
            raise ValueError("Unsupported save format. Use 'joblib' or 'pickle'.")
        
        logging.info(f'Model saved at {file_path}')

    
    def load_model(self, model_name, load_format='joblib', site='Penrose', pollutant='PM2.5'):
        """
        Load a saved model from disk.
    
        :param model_name: Name of the model.
        :param load_format: Format to load the model ('joblib' or 'pickle').
        :param site: Site information (e.g., 'Penrose' or 'Takapuna').
        :param pollutant: Pollutant information (e.g., 'PM2.5' or 'PM10').
        :return: Loaded model instance.
        """
        logging.info(f'Loading model {model_name} for site {site} and pollutant {pollutant}')
        file_extension = 'pkl' if load_format == 'pickle' else 'joblib'
        file_path = f"data/models/{site}_{pollutant}_{model_name}.{file_extension}"
        
        if load_format == 'joblib':
            model = joblib.load(file_path)
        elif load_format == 'pickle':
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
        elif load_format == 'pmml':
            raise NotImplementedError("PMML loading not implemented. Use a compatible library for your model type.")
        else:
            raise ValueError("Unsupported load format. Use 'joblib' or 'pickle'.")

        logging.info(f'Model loaded from {file_path}')
        return model

    # def generate_markdown_table(self, all_metrics, model_name, target_variable):
    #     """
    #     Generate a markdown table for cross-validation metrics.

    #     :param all_metrics: Dictionary of metrics with lists of scores for each fold.
    #     :param model_name: Name of the model.
    #     :param target_variable: Name of the target variable.
    #     :return: Markdown table as a string.
    #     """
    #     metrics = ['R2', 'RMSE', 'MSE', 'MAE', 'MAPE', 'Adjusted R2']
    #     header = f"| {target_variable} | {model_name} | Metric | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 |\n"
    #     header += "|---|---|---|---|---|---|---|---|\n"
    #     rows = []
    #     for metric in metrics:
    #         row = [f"{target_variable}", f"{model_name}", f"{metric}"]
    #         row.extend([f"{score:.2f}" for score in all_metrics[metric]])
    #         rows.append(" | ".join(row))
    #     return header + "\n".join(rows)

    def generate_markdown_table(self, target_variable, evaluation_results):
        """
        Generate a markdown table for cross-validation metrics for all models of a target variable.

        :param target_variable: Name of the target variable.
        :param evaluation_results: Dictionary of evaluation results for all models.
        :return: Markdown table as a string.
        """
        metrics = ['RMSE', 'MSE', 'MAE', 'MAPE', 'R2', 'Adjusted R2']
        header = "| Target | Model | Metric | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Training Time |\n"
        header += "|---|---|---|---|---|---|---|---|---|\n"
        rows = []
        for model_name, model_metrics in evaluation_results.items():
            for metric in metrics:
                row = [target_variable, model_name, metric]
                row.extend([f"{score:.2f}" for score in model_metrics[metric]])
                row.append(f"{model_metrics['Training Time']:.2f}")  ## Append total training time
                rows.append(" | ".join(row))
        return header + "\n".join(rows)
        
    ## Adjust the train_and_evaluate_models method to log markdown tables


#### Step 6: Train and Evaluate Models

## Train and evaluate models for each target variable
def train_and_evaluate_models(data_dict, model_names):
    """
    Train and evaluate models for each target variable in the data dictionary.

    :param data_dict: Dictionary containing training data, target variable, and feature list for each site and pollutant.

    :param model_names: List of model names to train and evaluate.

    :return: Dictionary of trained models and their evaluation results.
    """
    ## Initialize a dictionary to store trained models and evaluation results
    trained_models = {}
    evaluation_results = {}

    ## Train and evaluate models for each target variable
    for target_var, (train_data, target, features, arima_model) in data_dict.items():
        logging.info(f"\n🛠️ Training models for {target_var} ... \n")
        logging.info(f"Selected ARIMA model for {target_var}: {arima_model}")
        
        ## Initialize the PredictiveModels class for each target variable
        pm = PredictiveModels(train_data, target, features)

        ## Initialize dictionary for evaluation results of current target variable
        target_evaluation_results = {}
        
        for model_name in model_names:
            start_time = time.time()
            logging.debug(f"Fitting Model: {model_name} at {start_time}")
            if model_name == 'ARIMA':
                all_metrics = pm.adaptive_cross_validate(model_name, arima_model=arima_model, site=f'{target}', pollutant=f'{target_var}')
            else:
                all_metrics = pm.adaptive_cross_validate(model_name, site=f'{target}', pollutant=f'{target_var}')
            end_time = time.time()
            training_time = end_time - start_time
    
            # evaluation = pm.evaluate(model_name)
            # # pm.save_model(model_name, pm.models[model_name], site=target_var.split('_')[0], pollutant=target)
            # trained_models[f"{target_var}_{model_name}"] = pm.models[model_name]
            # target_evaluation_results[model_name] = evaluation
            
            all_metrics['Training Time'] = training_time
            target_evaluation_results[model_name] = all_metrics
            trained_models[f"{target_var}_{model_name}"] = pm.models[model_name]

        ## Store results for the current target variable
        evaluation_results[target_var] = target_evaluation_results

        ## Identify the best model for each target variable based on RMSE
        # best_model_name = min(target_evaluation_results.items(), key=lambda x: x[1]['RMSE'])[0]
        # best_model = trained_models[f"{target_var}_{best_model_name}"]  ## Key to access trained models
        ## select the model with the best average RMSE across all folds
        best_model_name = min(target_evaluation_results.items(), key=lambda x: np.mean(x[1]['RMSE']))[0]
        best_model = pm.models[best_model_name]

        logging.info(f"\nThe best model based on average RMSE across all folds for {target_var}: {best_model_name} \n")
        ## Generate and log the markdown table for all models and target variable
        markdown_table = pm.generate_markdown_table(target_var, target_evaluation_results)
        logging.info(f"\nMarkdown Table for {target_var}:\n{markdown_table}\n")
        
        ## Save the best model after evaluation for each target variable based on RMSE immediately after evaluation
        pm.save_model(best_model_name, best_model, site=target_var.split('_')[0], pollutant=target)

    return trained_models, evaluation_results


def generate_markdown_table(evaluation_results, folds=5):
    metrics = ['RMSE', 'MSE', 'MAE', 'MAPE', 'R2 Score','Adjusted R2']
    header = "| Target | Model | Metric | " + " | ".join([f"Fold{i+1}" for i in range(folds)]) + " |"
    separator = "|---" * (folds + 3) + "|"
    
    rows = []
    for target, models in evaluation_results.items():
        for model, results in models.items():
            for metric in metrics:
                row = [target, model, metric]
                row.extend([f"{results[metric]:.2f}" for fold in range(folds)])
                rows.append("| " + " | ".join(row) + " |")
    
    table = "\n".join([header, separator] + rows)
    return table


# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # 🧩 Predictive Models Development

# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 🛠️ Data Preparation

# In[11]:


def extract_feature_names(feature_list):
    return [feature[0] for feature in feature_list]

## Extracting feature names: target_variables = ['PM2.5', 'PM10'] across Penrose & Takapuna
top_features_data11_names = extract_feature_names(top_features_data11)
top_features_data12_names = extract_feature_names(top_features_data12)
top_features_data21_names = extract_feature_names(top_features_data21)
top_features_data22_names = extract_feature_names(top_features_data22)

## Combining data for easier access
data_dict = {
    'Penrose_PM2.5': (cleaned_data_site1, 'PM2.5', top_features_data11_names, 'ARIMA(0, 1, 4)'),
    'Takapuna_PM2.5': (cleaned_data_site2, 'PM2.5', top_features_data12_names, 'ARIMA(0, 1, 2)'),
    'Penrose_PM10': (cleaned_data_site1, 'PM10', top_features_data21_names, 'ARIMA(10, 0, 0)'),
    'Takapuna_PM10': (cleaned_data_site2, 'PM10', top_features_data22_names, 'ARIMA(2, 0, 3)')
}


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 🛠️ Model Development

# In[12]:


logging.getLogger('fbprophet').setLevel(logging.INFO)

## Define Models to Train
# model_names = ['ARIMA', 'Prophet', 'NeuralProphet', 'LinearRegression', 'RandomForest', 'SVR', 'XGBoost', 'LSTM', 'MLP']
model_names = ['ARIMA', 'Prophet', 'NeuralProphet', 'LinearRegression', 'Ridge', 'Lasso', 'RandomForest', 'SVR', 'XGBoost']

## Train and Evaluate Models
trained_models, evaluation_results = train_and_evaluate_models(data_dict, model_names)


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 🛠️ Save Models & Evaluation Results

# * [x] evaluation_results --> 'data/source/evaluation_results.json'
# * [ ] trained_models     --> select_best_model() --> best_models
# * [ ] best_models 

# In[46]:


trained_models

# evaluation_results


# In[45]:


import json

## Save the evaluation_results to a JSON file
with open('data/source/evaluation_results.json', 'w') as f:
    json.dump(evaluation_results, f)


# In[51]:


#### Step 7: Select the Best Model based on RMSE
def select_best_model(evaluation_results, key_metric='RMSE'):
    """
    Select the best model for each target variable .
    Select the best model for each target variable based on a specified key metric (by default, based on RMSE).

    Parameters:
    evaluation_results (dict): Dictionary containing evaluation metrics for each model and target variable.
    key_metric (str): The metric to be used as the key metric for selecting the best model.

    Returns:
    dict: Dictionary containing the best model for each target variable.
    """
    ## Identify the best model for each target variable
    best_models = {}
    for target_var, models in evaluation_results.items():
        try:
            logging.debug(f"Selecting best model for {target_var} based on {key_metric} from {models.items()}")
            ## Calculate the mean of the metric values across folds
            metric_values = {model_name: np.mean(metrics.get(key_metric, [float('inf')])) for model_name, metrics in models.items()}
            # best_model = min(models.items(), key=lambda x: np.mean(x[1].get('RMSE', float('inf'))))[0]
            metric_values = {model_name: np.mean(metrics.get(key_metric, [float('inf')])) for model_name, metrics in models.items()}
            if key_metric in ['RMSE', 'MSE', 'MAE', 'MAPE', 'Training Time']: ## Lower value is better
                best_model = min(metric_values, key=metric_values.get)
            else:
                best_model = max(metric_values, key=metric_values.get)
            best_models[target_var] = best_model
            # logging.info(f"The best model for {target_var} is {best_model} with RMSE: {models[best_model]['RMSE']}")
            logging.info(f"The best model for {target_var} based on {key_metric} is {best_model} with {key_metric}: {metric_values[best_model]}")
        except (IndexError, ValueError, KeyError) as e:
            logging.error(f"Error selecting best model for {target_var}: {e}")
    return best_models


## Define key metrics to evaluate
# key_metrics = ['RMSE', 'MSE', 'MAE', 'R2', 'Adjusted R2', 'Training Time']
key_metrics = ['RMSE', 'MSE', 'MAE']
## Get best models for each key metric
best_models_per_metric = {metric: select_best_model(evaluation_results, key_metric=metric) for metric in key_metrics}

# ## [DEBUG] Save the best models to disk
# for target_var, model_name in best_models.items():
#     model_filename = f"data/models/{target_var}_{model_name}.joblib"
#     ## Save the model (assuming the model object is available in the current scope)
#     # pickle.dump(models[model_name], open(model_filename, 'wb'))  ## Uncomment this line when model objects are available
#     logging.info(f"Based on RMSE, Model {model_name} for {target_var} saved at {model_filename}")


# In[53]:


## Print the results
for metric, best_models in best_models_per_metric.items():
    logging.info(f"\n\nBest Models based on {metric}:\n")
    for target_var, model_name in best_models.items():
        logging.info(f"Target Variable: {target_var}, Best Model: {model_name}")


# In[54]:


def generate_markdown_table(evaluation_results):
    """
    Generate a markdown table summarizing the best model results across multiple metrics for each target variable.

    Parameters:
    evaluation_results (dict): Dictionary containing evaluation metrics for each model and target variable.

    Returns:
    str: Markdown table as a string.
    """
    table = "| Target Variable | Best Model | RMSE | MSE | MAE | R2 | Adjusted R2 | Training Time |\n"
    table += "|-----------------|------------|------|-----|-----|----|-------------|---------------|\n"

    # best_models = select_best_model(evaluation_results)
    best_models = best_models_per_metric['RMSE']
    for target_var, best_model in best_models.items():
        metrics = evaluation_results[target_var][best_model]
        table += (f"| {target_var} | {best_model} | {np.mean(metrics['RMSE']):.3f} | {np.mean(metrics['MSE']):.3f} | "
                  f"{np.mean(metrics['MAE']):.3f} | {np.mean(metrics['R2']):.3f} | {np.mean(metrics['Adjusted R2']):.3f} | "
                  f"{np.mean(metrics['Training Time']):.3f} |\n")
    
    return table

# Generate and log the markdown table
markdown_table = generate_markdown_table(evaluation_results)
logging.info(f"\nMarkdown Table of Best Models:\n{markdown_table}\n")


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 🛠️ Model Performance Visualization

# In[58]:


def parse_evaluation_results(evaluation_results):
    """
    Parses the evaluation results into a DataFrame.
    
    Args:
        evaluation_results (dict): Dictionary containing model evaluation results.
        
    Returns:
        pd.DataFrame: Parsed data in a DataFrame.
    """
    logging.info("Parsing evaluation results into a DataFrame.")
    rows = []
    for target, models in evaluation_results.items():
        for model, metrics in models.items():
            for metric, values in metrics.items():
                if isinstance(values, list):
                    for fold, value in enumerate(values, start=1):
                        rows.append({"Target": target, "Model": model, "Metric": metric, "Fold": f"Fold{fold}", "Value": value})
                else:
                    rows.append({"Target": target, "Model": model, "Metric": metric, "Fold": "Training Time", "Value": values})
    df = pd.DataFrame(rows)
    logging.info("Finished parsing evaluation results.")
    return df


# In[59]:


## @depreciated: Save evaluation results to a .json file --> Parse the evaluation results into a DataFrame
# evaluation_df = pd.DataFrame(evaluation_results).transpose()
# evaluation_df.to_csv('data/source/evaluation_results.csv', index=True)
# logging.info(f"Evaluation results saved to evaluation_results.csv")


# In[60]:


# trained_models
# evaluation_results

## Parse the evaluation results into a DataFrame
df = parse_evaluation_results(evaluation_results)
# df

## Save the DataFrame to a CSV or Parquet file or a pickle file
df.to_csv('data/source/evaluation_results.csv', index=False)
# df.to_pickle('evaluation_results_df.pkl')
# df.to_parquet('evaluation_results_df.parquet')


# In[61]:


## If using pickle file or Parquet
# df = pd.read_pickle('evaluation_results_df.pkl')
# df = pd.read_parquet('evaluation_results_df.parquet')
## Load the DataFrame from the *.csv file
df = pd.read_csv('data/source/evaluation_results.csv')

## Proceed with EDA
print(df.head())


# In[62]:


import plotly.express as px
import plotly.graph_objects as go

def create_visualization(df, default_metric='RMSE'):
    """
    Creates a polar bar plot visualization for model performance comparison.
    
    Args:
        df (pd.DataFrame): DataFrame containing parsed evaluation results.
        default_metric (str): The default metric to be displayed in the polar plot.
    """
    logging.debug(f"Creating the polar bar plot visualization for metric: {default_metric}")
    
    targets = df['Target'].unique()
    models = df['Model'].unique()
    metrics = df['Metric'].unique()
    colorscale = px.colors.sequential.Plasma

    fig = go.Figure()

    ## Iterate over each target: Add traces for the default metric
    for target in targets:
        ## Filter DataFrame for the current target and default metric
        target_df = df[(df['Target'] == target) & (df['Metric'] == default_metric)]
        for model in models:
            model_df = target_df[target_df['Model'] == model]
            mean_value = model_df['Value'].mean()
            # mean_value = model_df.groupby('Model')['Value'].mean().values
            if len(model_df) > 0:
                # hovertext = [f"Mean {default_metric}: {mean_value:.2f}"] + model_df.apply(lambda row: f"Fold {row['Fold']}: {row['Value']:.2f}", axis=1).tolist()
                # logging.debug(hovertext)
                
                fig.add_trace(
                    go.Barpolar(
                        r=[mean_value],                ## Average/Mean value for the metric
                        theta=[model],                 ## Display the model names around the polar chart
                        name=f"{target} - {model}",    ## Only target in the name for legend clarity
                        legendgroup=target,
                        showlegend=model == models[0], ## Only show legend for the first model to avoid repetition
                        # text=hovertext,                ## Hover text for additional info
                        # hoverinfo='text',
                        # text=model_df.apply(lambda row: f"{row['Model']} ({row['Fold']}): {row['Value']}", axis=1),
                        text = [
                            f"Model: {model}<br>"
                            f"Metric: {default_metric}<br>"
                            f"Average {default_metric}: {mean_value:.2f}<br>" +
                            "<br>".join([f"Fold {i+1}: {model_df.iloc[i]['Value']:.2f}" for i in range(len(model_df))])
                        ],
                        hoverinfo='text+r',
                    )
                )

    
    ## Set up the layout with 2/3 for the polar chart and 1/3 for the dropdown and legend
    fig.update_layout(
        title="Comparative Model Performance Across Multiple Metrics for Penrose and Takapuna PM2.5 and PM10",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, df[df['Metric'] == default_metric]['Value'].max()])
        ),
        showlegend=True,
        # template="plotly_dark",
        # legend=dict(yanchor="top", y=1, xanchor="left", x=1.35),
        legend=dict(
            title="Targets",
            itemsizing='constant',
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.2,
            font=dict(size=10),  # Adjust font size for better readability
            bgcolor="rgba(255,255,255,0.7)"  # Add a semi-transparent background for clarity
        ),
        margin=dict(l=60, r=30, t=40, b=30),
        width=1200,  ## Adjust width to allow space for dropdown and legend
        height=800,  ## Adjust height for better layout
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": metric,
                        "method": "update",
                        "args": [
                            {
                                # "visible": [
                                #     # trace.name.split(' - ')[0] == target and trace.name.split(' - ')[1] in models
                                #     # for trace in fig.data
                                #     # for target in targets
                                #     True  ## Ensure all traces remain visible
                                #     for trace in fig.data
                                # ]
                                "visible": [True for _ in fig.data]
                            },
                            {"title": f"Comparative Model Performance for {metric}"},
                            {
                                "showlegend": True
                            }
                        ]
                    }
                    for metric in metrics
                ],
                "direction": "down",
                "showactive": True,
                "xanchor": "left",
                "x": 0.01,
                "y": 1.2,
            }
        ],
        # autosize=False,  ## Ensure layout respects the specified width and height
    )

    ## FIXME: Set visibility of traces: also show remain 3 target variables but deselect them
    # for trace in fig.data:
    #     trace.visible = (trace.name.split(' - ')[0] == targets[0])

    ## Ensure colors are unique by rounding values to zero/two decimal places --> converting rounded values to distinct integers
    decimal_place = 0
    rounded_values = sorted({round(v, decimal_place) for v in df[df['Metric'] == default_metric]['Value']})
    unique_colors = {v: i for i, v in enumerate(rounded_values)}
    logging.debug(f"Unique colors mapping: {unique_colors}")

    ## Map the value to a color in the Plasma colorscale
    for trace in fig.data:
        value = round(trace.r[0], decimal_place)
        if value in unique_colors:
            color_index = unique_colors[value]
        else:
            color_index = min(unique_colors.values(), key=lambda k: abs(k - value))
        color_index = max(0, min(color_index, len(colorscale) - 1))
        trace.marker.color = colorscale[color_index]
        trace.marker.colorscale = colorscale  ## Apply gradient scale
        trace.marker.showscale = True         ## Ensure gradient scale is shown
        # trace.marker.colorbar = dict(title='Value')
        trace.marker.colorbar = dict(title=f'{default_metric} Value')
        logging.debug(f"Value: {value}, Color Index: {color_index}, Color: {colorscale[color_index]}")

    ## Using a color scale to set colors properly
    # fig.update_traces(marker=dict(colorscale='Plasma', showscale=True))
    ## Update traces with color scale and color bar
    fig.update_traces(
        marker=dict(
            colorbar=dict(title=f'{default_metric} Value')
        )
    )

    fig.show()
    logging.debug("Visualization created successfully.")


create_visualization(df)


# <footer style="padding-bottom:35px; background:#f9f9f9; border-bottom:3px solid #00b2b1">
#     <div style="float:left;margin-top:14px;color:#E37C4D">Predicting Air Particulate Matter at Scale</div>
#     <div style="float:right;">
#         <div style="float:left; margin-top:14px">
#             Auckland University of Technology (AUT)
#         </div>
#     </div>
# </footer>
