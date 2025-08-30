#!/usr/bin/env python
# coding: utf-8

# <header style="padding:1px;background:#f9f9f9;border-top:3px solid #00b2b1"><img id="Teradata-logo" src="https://www.teradata.com/Teradata/Images/Rebrand/Teradata_logo-two_color.png" alt="Teradata" width="220" align="right" />
# 
# <b style='font-size:28px;font-family:Arial;color:#E37C4D'>🎓 Predicting Air Particulate Matter at Scale ⛅️</b><br>
# <b style='font-size:28px;font-family:Arial;color:#E37C4D'>🛠️ The Visual Advanced Analytics Dashboard 📊</b>
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
# <p>🎓 The Visual Analytics Dashboard 📊 in action, take a look at the  live demo running at <a href="https://analytics-experience-28038b544779.herokuapp.com/" target="_blank">Analytics-Experience Dashboard ⛅️</a>
#     
# 
# **Workflow Steps:**
# 
# 1. Libraries and Reusable Functions
# 2. Data to Viz
# 
# </div>

# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
#     
# # 🎯 Libraries and Reusable Functions 

# In[1]:


import pandas as pd              ## Data processing, file I/O
import numpy as np               ## Linear algebra

# import plotly.express as px
import plotly.graph_objects as go

import vizro.models as vm
import vizro.plotly.express as px
from vizro import Vizro
from vizro.tables import dash_ag_grid
from vizro.models.types import capture


# <div class="alert alert-block alert-info">
# 🎓 To execute all of the cells in <b>DataFrameAdapter.ipynb</b>.
# </div>

# In[2]:


import os, logging
from dotenv import load_dotenv

# %run -i ./Data_Loading_and_Descriptive_Statistics.ipynb
from Data_Loading_and_Descriptive_Statistics import load_environment_settings, load_and_prepare_data_from_files, display_site_comparison, DescriptiveStatistics

## Python usage
# if __name__ == "__main__":
#     load_environment_settings(env_path_default='cleaned_vantage.env')

## .env --> Setting the environment variable for output format programmatically
env_settings = load_environment_settings(env_path_default='cleaned_dashboard.env', is_print_table=False)

IS_JUPYTERLAB               = env_settings['IS_JUPYTERLAB']
IS_LOADING_FROM_FILES       = env_settings['IS_LOADING_FROM_FILES']
IS_TERADATA_VANTAGE         = env_settings['IS_TERADATA_VANTAGE']
IS_DATA_IN_TERADATA_VANTAGE = env_settings['IS_DATA_IN_TERADATA_VANTAGE']
IS_TEST_DEV                 = env_settings['IS_TEST_DEV']
USE_DATA_PREFIX             = env_settings['USE_DATA_PREFIX']
SCHEMA_NAME                 = env_settings['SCHEMA_NAME']
TABLE_NAME                  = env_settings['TABLE_NAME']
DATA_PATH                   = f'data/{USE_DATA_PREFIX}'   ## 'raw' | 'cleaned'


# In[3]:


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

## Display the first few rows of the final dataframe
# rawdata.head().style.set_properties(subset=['Timestamp'], **{'background-color': 'dodgerblue'})
# print("\nℹ️ The Shape of the rawdata Dataframe:", rawdata.shape)

rawdata


# In[4]:


# ## The rawdata is retrieved from "2.1. Load Data from *.csv files" and/or "2.2. Prepare Data in Teradata Vantage"
# rawdata_site1 = rawdata[rawdata['Site'] == 'Penrose'][include_columns_site1]
# rawdata_site2 = rawdata[rawdata['Site'] == 'Takapuna'][include_columns_site2]
# print("ℹ️ The Shape of the Dataframe rawdata_site1 (Penrose) and rawdata_site2 (Takapuna):", rawdata_site1.shape, rawdata_site2.shape)
# # logging.info("ℹ️ The Shape of the DataFrame rawdata_site1 (Penrose) and rawdata_site2 (Takapuna): %s, %s", rawdata_site1.shape, rawdata_site2.shape)

# ## Call the function to display the table
# print('\n🎓 [Data_Loading_and_Descriptive_Statistics.ipynb] Listing variables with description...')
# display_site_comparison()


# In[5]:


# %run -i ./DataFrameAdapter.ipynb
from DataFrameAdapter import DataFrameAdapter, display_cleaned_site_comparison

# cleaned_data = DataFrameAdapter.engineer_features(rawdata.copy())
cleaned_data = DataFrameAdapter.engineer_features(rawdata)
# cleaned_data

## Describe Data and attribute-types: numerical(continuous), categorical (nominal, ordinal) ==> 
## 'Timestamp' as the only Ordinal Attribute/Column given its nature order in time-series data
cleaned_ordinal_columns = ['Timestamp']

print("\n🎓 Describing the types of each attribute as cleaned_numerical_columns (Continuous), cleaned_ordinal_columns (Ordinal), or cleaned_nominal_columns (Nominal) ...")
cleaned_numerical_columns, cleaned_nominal_columns = DescriptiveStatistics.describe_data(cleaned_data, cleaned_ordinal_columns)

cleaned_data_site1 = cleaned_data[cleaned_data['Site'] == 'Penrose']
cleaned_data_site2 = cleaned_data[cleaned_data['Site'] == 'Takapuna']

# if IS_DEBUG:
print("\n🎓 [Site1 - Penrose]  Summary Statistics of the {site1} cleaned_data_site1 Dataframe such as the mean, max/minimum values ...")
cleaned_data_site1.describe()   

# if IS_DEBUG:
print("\n🎓 [Site2 - Takapuna]  Summary Statistics of the {site2} cleaned_data_site2 Dataframe such as the mean, max/minimum values ...")
cleaned_data_site2.describe()

# target_variables = {'PM2.5': 'Particulate Matter <2.5 µm', 'PM10': 'Particulate Matter <10 µm'}
# data_series_df1 = PandasDataLoader.load_data(rawdata_site1)
# data_series_df2 = PandasDataLoader.load_data(rawdata_site2)
# data_series_df1

## Call the function to display the table
print('\n🎓 [DataFrameAdapter.ipynb] Listing variables with description...')
display_cleaned_site_comparison(is_include_rawdata=False)


# In[6]:


df_data_site1 = cleaned_data_site1.copy()

# print("\n🎓 Summary Statistics of the df_data_site1 Dataframe such as the mean, maximum and minimum values ...")
# df_data_site1.describe()

df_data_site1


# In[7]:


df_data_site2 = cleaned_data_site2.copy()

# print("\n🎓 Summary Statistics of the df_data_site1 Dataframe such as the mean, maximum and minimum values ...")
# df_data_site2.describe()

df_data_site2


# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # 📊 Data to Viz

# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [Data to Viz] 1. Home Page

# In[8]:


## Load dataset from a specified path and ensure it's loaded correctly with a predefined function
monitoring_sites_metadata_file_path = 'data/source/Air-Quality-Monitoring-Network.csv'
monitoring_sites_metadata_df = pd.read_csv(monitoring_sites_metadata_file_path)
# monitoring_sites_metadata_df

## Load dataset from a specified path. 
## Source: WHO Ambient Air Quality Database (Update Jan 2024)at https://www.who.int/data/gho/data/themes/air-pollution/who-air-quality-database
aap_metadata_file_path = 'data/source/who_ambient_air_quality_database_version_2024_v6.1.csv'
aap_metadata_df = pd.read_csv(aap_metadata_file_path)
aap_metadata_df['Measurement Year'] = aap_metadata_df['Measurement Year'].fillna(0).astype(int)  ## Convert to int, handle NaN if necessary
aap_metadata_df


@capture("graph")
def variable_map(data_frame: pd.DataFrame, location_col: str, color_col: str, hover_col: str, frame_col: str, title: str, scope: str = "world"):
    """
    Generates a choropleth map for visualizing geographical data over time.
    
    Args:
    - data_frame (pd.DataFrame): DataFrame containing the data.
    - location_col (str): Column in DataFrame that denotes geographic locations (ISO codes).
    - color_col (str): Column used to determine the color of each region, representing different values.
    - hover_col (str): Column that will appear in the tooltip when hovering over an area.
    - frame_col (str): Column for animation frame (typically years).
    - title (str): Title of the map.
    - scope (str): Geographic scope of the map ('world', 'usa', 'europe', etc.).
    - lat_col (str): Optional latitude column for precise location mapping.
    - lon_col (str): Optional longitude column for precise location mapping.

    Returns:
    - plotly.graph_objects.Figure: Interactive choropleth map.
    """
    # Determine if geographic coordinates are present
    # if 'latitude' in data_frame.columns and 'longitude' in data_frame.columns:
    fig = px.choropleth(
        data_frame,
        locations=location_col,
        ## Use latitude and longitude for plotting points if provided
        # lat='latitude',
        # lon='longitude',
        color=color_col,
        ## color_continuous_scale = px.colors.sequential.Plasma | 'OrRd' | 'Viridis'
        color_continuous_scale='OrRd', 
        hover_name=hover_col,
        hover_data={   ## Include more context in hover tooltips
            'City': True, 'Population': True, 
            # 'Latitude': True, 'Longitude': True
        },
        animation_frame=frame_col,
        labels={
            # frame_col: frame_col.capitalize(),
            # color_col: color_col.replace('_', ' ').capitalize()
            frame_col: frame_col,
            color_col: color_col.replace('_', ' ')
        },
        title=title,
        scope=scope
    )
    
    # fig.update_layout(showlegend=True)
    fig.update_layout(showlegend=False)
    # fig.update_geos(fitbounds="locations") ## Ensures the map is centered and zoomed appropriately to cover the relevant data points.
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    fig.update_coloraxes(colorbar={"thickness": 10, "title": {"side": "right"}})
    return fig

def create_home_page():
    """Function returns the Home page."""

    tab_1 = vm.Container(
        title="💹 Visual Analytics",
        layout=vm.Layout(grid=[[0, 1], [2, 3]], row_gap="18px", col_gap="18px"),
        components=[
            vm.Card(
                text="""
                    ### 📈 Time-Series Analysis

                    📈 1. Line Plot with selector over Hours, Days, Months, Years

                    📈 2. Scatter Plot Time Series by Hour of Day
                    
                    📈 3. Line Plot: Hour of the Day per Every Day of the Week 
                    
                    📈 4. Box Plot: Distribution by the Hour in each Season
                    
                    📈 5. Bar Plot: Total PM Concentration per Month/Season/Year

                """,
                href="/pm25-variable-analysis-for-penrose",
            ),
            vm.Card(
                text="""
                        ### 📊 **RQ1.** Feature Correlation Sankey Diagram

                        > **RQ1:** What are the insights (emerging trends, seasonal patterns) from historical data on PM2.5/PM10 levels, and the influence of environmental data on PM2.5/PM10 levels in Auckland? \

                        &nbsp;
                        
                        Discover emerging trends and seasonal patterns in time series data.

                        Investigating the **interconnection** between pollutants and meteorological/weather features/variables.
                    """,
                href="/feature-correlation-visualization",
            ),
            vm.Card(
                text="""
                    ### 💹 **RQ3.** Visual Analytics Dashboard

                    > **RQ3:** How can we effectively visualise the analysis insights of historical PM2.5/PM10 levels and effectively communicate predictive outcomes to stakeholders and the public in an easy-to-understand, interpretable manner? 
                    
                    &nbsp;
                    
                    🆓 **Data to Visualisation** using Open-Source Visual Analytics.
                    
                    🎓 **Executive Summary**: Summarizing the main findings for each location.
                """,
                href="/executive-summary",
            ),
            vm.Card(
                text="""
                    ### 🧠 **RQ2.** Predictive Analytics Models/Algorithms

                    > **RQ2:** What _time series_, machine learning and deep learning models offer the best performance to predict PM2.5/PM10 levels based on data from the Environmental Data in Auckland?
                    
                    &nbsp;
                    
                    🌟 Comparative Benchmark Analysis: Discovering how the metrics differ for each location compared to WHO air quality guidelines and export data for further investigation.
                """,
                href="/predictive-analytics-models-and-algorithms",
            ),
        ],
    )

    ##Create a customized AgGrid display with conditional formatting based on environmental monitoring data.
    ## Define custom cell style for 'Site Name' based on conditions
    cellStyle = {
        "styleConditions": [
            {
                "condition": "params.node.data['Is Meteorological'] === 'NO'",
                "style": {"color": "blue"},
            },
            {
                "condition": "params.node.data['PM25 Monitored'] === 'NO'",
                "style": {"color": "green"},
            },
            {
                "condition": "params.node.data['Is Meteorological'] === 'YES' && params.node.data['PM25 Monitored'] === 'YES'",
                "style": {"color": "orange"},
            },
        ]
    }
    ## Define column properties, highlight 'Site Name'
    columnDefs = [
        {"field": "Site ID", "cellStyle": cellStyle},
        {"field": "Site Name", "cellStyle": cellStyle},
        # {"field": "Country"},
        # {"field": "City"},
        {"field": "Pollutants Monitored", "cellStyle": cellStyle},
        # {"field": "PM2.5 Monitored"},
        {"field": "Meteorological Measured", "cellStyle": cellStyle},
        # {"field": "Is Meteorological"},
        {"field": "Site Class", "cellStyle": cellStyle},
        # {"field": "latitude"},
        # {"field": "longitude"},
        # {"field": "Site Address"},
        {"field": "Established Date", "cellStyle": cellStyle},
    ]
    
    tab_2 = vm.Container(
        title="📂 Environmental Dataset",
        layout=vm.Layout(grid=[[0],[1]], row_min_height="550px"),
        # layout=vm.Layout(grid=[[0,0,1],[2,2,2]], row_min_height="450px"),
        components=[
            vm.Graph(
                id="variable_map",
                figure=variable_map(
                    data_frame=aap_metadata_df,
                    location_col="ISO3",
                    # location_col="City",
                    color_col="PM2.5 Concentration (µg/m³)",
                    hover_col="Country Name",
                    # hover_col="City",
                    frame_col="Measurement Year",
                    title="WHO Air Quality Trends by Country Over Time",
                    scope="world"  ## Adjust 'scope' as needed ('europe', 'asia', 'north america', etc.)
                ),
            ),
            vm.AgGrid(figure=dash_ag_grid(data_frame=monitoring_sites_metadata_df, 
                                          columnSize="sizeToFit",
                                          columnDefs=columnDefs)),
        ],
        # controls=[
        #     vm.Parameter(
        #         targets=["variable_map.color"],
        #         selector=vm.RadioItems(options=["PM2.5 Concentration (µg/m³)", "PM10 Concentration (µg/m³)"], title="Select variable"),
        #     )
        # ],
    )
    
    # tab_3 = vm.Container(
    #     title="⚙️ Agile CRISP-DM",
    #     components=[
    #         vm.Card(
    #             text="""
    #                 ![Agile-implementation-of-CRISP-DM](https://analytics-experience.pages.dev/assets/images/Agile-implementation-of-CRISP-DM-5711e3224a2bea3c5bc33b0bafbb5b43.gif)
                    
    #                 🗓️ Project Timeline/Schedule & Deliverables 🚀: https://analytics-experience-calendar.onrender.com 
    #             """
    #         ),
    #     ],
    # )

    # tab_4 = vm.Container(
    #     title="🛠️ Enterprise at Scale",
    #     components=[
    #         vm.Card(
    #             text="""
    #                 ![🛠️ Advanced Analytics & Machine Learning at Scale ⛅️](https://analytics-experience.pages.dev/assets/images/Advanced-Analytics-Machine-Learning-at-Scale-a64ec9faad408f3deb86225de7b3fbee.gif)
    #             """
    #         ),
    #     ],
    # )

    tab_5 = vm.Container(
        title="🌏 Documentation",
        components=[
            vm.Card(
                text="""
                    # 🛠️ Advanced Analytics & Machine Learning at Scale ⛅️

                    ## 🔎 Research Project: Predicting Air Particulate Matter at Scale

                    ---
                    
                    ### 🗓️ Project Timeline/Schedule & Deliverables 🚀
                    
                    Please refer to https://analytics-experience-calendar.onrender.com
                    
                    ### 🛠️ An Agile implementation of CRISP-DM
                    
                    Please refer to https://analytics-experience.pages.dev/docs/data-science/project-roadmap
                    
                    ### ⛅️ Analytics & Machine Learning at Scale
                    
                    Please refer to https://analytics-experience.pages.dev/docs/data-science/project-proposal

                    ---

                    > ✍️ I would like to express my gratitude to **Dr. Nuttanan Wichitaksorn** as my professor and supervisor, **Dr. Victor Miranda** as STAT995 course leader and coordinator at Auckland University of Technology (**AUT**), **Jason Sharpe** as my supervisor, and also to **Douglas H. Ebel** and **Bokareva Tatiana** from **Teradata®** for their great support 🙏.
                """
            ),
        ],
    )

    ### BACKUP: markdown &nbsp;
    # ![](assets/images/icons/line-chart.svg#icon-top)
    # ![](assets/images/icons/hypotheses.svg#icon-top)
    # ![](assets/images/icons/collections.svg#icon-top)
    # ![](assets/images/icons/features.svg#icon-top)
    # * [🛠️ An Agile implementation of CRISP-DM](https://analytics-experience.pages.dev/docs/data-science/project-roadmap){:target="_blank"}  
    # * [⛅️ Advanced Analytics & Machine Learning at Scale](https://analytics-experience.pages.dev/docs/data-science/project-proposal) {:target="_blank"}  
    # ![Agile-implementation-of-CRISP-DM](https://analytics-experience.pages.dev/assets/images/Agile-implementation-of-CRISP-DM-5711e3224a2bea3c5bc33b0bafbb5b43.gif)
    #  🗓️ Project Timeline/Schedule & Deliverables 🚀: https://analytics-experience-calendar.onrender.com 
    #  ![🛠️ Advanced Analytics & Machine Learning at Scale ⛅️](https://analytics-experience.pages.dev/assets/images/Advanced-Analytics-Machine-Learning-at-Scale-a64ec9faad408f3deb86225de7b3fbee.gif)

    page_home = vm.Page(
        title="Home",
        # description="Intelligence Dashboard for Analytics-Experience project.",
        description="[Research Project] Predicting Air Particulate Matter at Scale.",
        # components=[vm.Tabs(tabs=[tab_1, tab_2, tab_3, tab_4, tab_5])], 
        components=[vm.Tabs(tabs=[tab_1, tab_2, tab_5])], 
                   # controls=[
                   #     # vm.Filter(column='Site', selector=vm.Dropdown(value=['ALL'])),
                   #     vm.Filter(column='Site', selector=vm.Dropdown(value="Penrose", multi=False, title="Select Location")),
                   # ],
        )

    return page_home


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [Data to Viz] 2. Data Preparation Page

# In[9]:


from typing import List, Literal
import dash_bootstrap_components as dbc
import vizro.models as vm
from dash import html
from vizro import Vizro

try:
    from pydantic.v1 import Field, PrivateAttr
except ImportError:
    from pydantic import PrivateAttr

from vizro.models import Action
from vizro.models._action._actions_chain import _action_validator_factory
from vizro.models.types import capture

## Carousel.1. Create new custom component
class Carousel(vm.VizroBaseModel):
    type: Literal["carousel"] = "carousel"
    items: List[dict] = []  ## List of items each with keys: 'key', 'src', 'text'
    actions: List[vm.Action] = []  ## Actions that can be triggered
    controls: bool = True
    indicators: bool = True

    _set_actions = _action_validator_factory("active_index")  

    def build(self):
        ## Returns a dbc.Carousel with items and controls based on class attributes
        return dbc.Carousel(
            id=self.id,
            items=[{"key": item['key'], "src": item['src'], "caption": item.get('text', '')} for item in self.items],
            # controls=self.controls,     ## Enables navigation controls ?
            # indicators=self.indicators, ## Enables navigation indicators ?
            # style={'height': '300px'}  # Ensures carousel height is maintained at 500px
        )

# ## Define the page with the custom Carousel component
# def create_carousel_page(items: List[dict], title: str) -> vm.Page:
#     """Generates a page with a carousel displaying images and descriptions/texts."""
#     # ## Carousel.3. Create custom action
#     # @capture("action")
#     # def handle_carousel_change(active_index: int) -> str:
#     #     """Handles carousel slide change to display corresponding slide number."""
#     #     ## Proper function definition and return based on active_index
#     #     return f"Slide number: {active_index}"

#     carousel = Carousel(
#         id="my-carousel",
#         items=items,
#         # actions=[
#         #     vm.Action(
#         #         function=handle_carousel_change(),
#         #         inputs=["my-carousel.active_index"],
#         #         outputs=["carousel-text-card.children"],
#         #     )
#         # ]
#     )
    
#     # card_text = vm.Card(text=items[0]['text'], id="carousel-text-card") ## Default to first item's text

#     ## Define a page with the custom Carousel component
#     return vm.Page(
#         title=title,
#         # layout=vm.Layout(grid=[[i] for i in range(2)], row_min_height="500px"),
#         layout=vm.Layout(grid=[[i] for i in range(1)], row_min_height="500px"),
#         components=[
#             # card_text,
#             carousel,
#         ],
#     )
# page_descriptive_statistics = create_carousel_page(items_descriptive_statistics, "Correlations Heatmap with Regression")


items_missing_values_imputation_penrose = [
    {"key": "1", "src": "assets/images/missing-values-imputation/items_missing_values_imputation_penrose_1.png", "text": "[Penrose][Historical Average Imputation] PM2.5"},
    {"key": "2", "src": "assets/images/missing-values-imputation/items_missing_values_imputation_penrose_2.png", "text": "[Penrose][Historical Average Imputation] PM2.5"},
    {"key": "3", "src": "assets/images/missing-values-imputation/items_missing_values_imputation_penrose_3.png", "text": "[Penrose][Historical Average Imputation] PM10"},
    {"key": "4", "src": "assets/images/missing-values-imputation/items_missing_values_imputation_penrose_4.png", "text": "[Penrose][Historical Average Imputation] PM10"},
    {"key": "6", "src": "assets/images/missing-values-imputation/items_missing_values_imputation_penrose_6.png", "text": "[Penrose][1-WEEK Moving Average Imputation] PM2.5"},
    {"key": "8", "src": "assets/images/missing-values-imputation/items_missing_values_imputation_penrose_8.png", "text": "[Penrose][1-DAY Moving Average Imputation] PM2.5"},
    {"key": "5", "src": "assets/images/missing-values-imputation/items_missing_values_imputation_penrose_5.png", "text": "[Penrose][1-WEEK Moving Average Imputation] PM10"},
    {"key": "7", "src": "assets/images/missing-values-imputation/items_missing_values_imputation_penrose_7.png", "text": "[Penrose][1-DAY Moving Average Imputation] PM10"},
]

items_descriptive_statistics = [
    {"key": "1", "src": "assets/images/Descriptive-Statistics/correlations_heatmap_with_regression-1.png", "text": "First Image Description"},
    {"key": "2", "src": "assets/images/Descriptive-Statistics/correlations_heatmap_with_regression-2.png", "text": "Second Image Description"},
    {"key": "3", "src": "assets/images/Descriptive-Statistics/correlation-matrix-heatmap-penrose.png", "text": "[Site1 - Penrose] Correlation Matrix Heatmap"},
    {"key": "4", "src": "assets/images/Descriptive-Statistics/correlation-matrix-heatmap-takapuna.png", "text": "[Site2 - Takapuna] Correlation Matrix Heatmap"},
]



# @capture("graph")
# def variable_map(data_frame: pd.DataFrame, location_col: str, color_col: str, hover_col: str, frame_col: str, title: str, scope: str = "world"):
#     """
#     Generates a choropleth map for visualizing geographical data over time.
    
#     Args:
#     - data_frame (pd.DataFrame): DataFrame containing the data.
#     - location_col (str): Column in DataFrame that denotes geographic locations (ISO codes).
#     - color_col (str): Column used to determine the color of each region, representing different values.
#     - hover_col (str): Column that will appear in the tooltip when hovering over an area.
#     - frame_col (str): Column for animation frame (typically years).
#     - title (str): Title of the map.
#     - scope (str): Geographic scope of the map ('world', 'usa', 'europe', etc.).
#     - lat_col (str): Optional latitude column for precise location mapping.
#     - lon_col (str): Optional longitude column for precise location mapping.

#     Returns:
#     - plotly.graph_objects.Figure: Interactive choropleth map.
#     """
#     # Determine if geographic coordinates are present
#     # if 'latitude' in data_frame.columns and 'longitude' in data_frame.columns:
#     fig = px.choropleth(
#         data_frame,
#         locations=location_col,
#         ## Use latitude and longitude for plotting points if provided
#         # lat='latitude',
#         # lon='longitude',
#         color=color_col,
#         ## color_continuous_scale = px.colors.sequential.Plasma | 'OrRd' | 'Viridis'
#         color_continuous_scale='OrRd', 
#         hover_name=hover_col,
#         hover_data={   ## Include more context in hover tooltips
#             'City': True, 'Population': True, 
#             # 'Latitude': True, 'Longitude': True
#         },
#         animation_frame=frame_col,
#         labels={
#             # frame_col: frame_col.capitalize(),
#             # color_col: color_col.replace('_', ' ').capitalize()
#             frame_col: frame_col,
#             color_col: color_col.replace('_', ' ')
#         },
#         title=title,
#         scope=scope
#     )
    
#     # fig.update_layout(showlegend=True)
#     fig.update_layout(showlegend=False)
#     # fig.update_geos(fitbounds="locations") ## Ensures the map is centered and zoomed appropriately to cover the relevant data points.
#     fig.update_yaxes(automargin=True)
#     fig.update_xaxes(automargin=True)
#     fig.update_coloraxes(colorbar={"thickness": 10, "title": {"side": "right"}})
#     return fig

def create_data_preparation_page():
    """Function returns the Data Preparation page."""

    ## Carousel.2. Add new components to expected type - here the selector of the parent components
    vm.Page.add_type("components", Carousel)
    vm.Container.add_type("components", Carousel)

    tab_1 = vm.Container(
        title="1️⃣ [Penrose] Missing Values Imputation",
        components=[
            Carousel(
                id="missing-values-imputation-carousel-penrose",
                items=items_missing_values_imputation_penrose,
            ),
        ],
    )

    tab_2 = vm.Container(
        title="🔎 Correlations Heatmap with Regression",
        components=[
            Carousel(
                id="carousel-descriptive-statistics",
                items=items_descriptive_statistics,
            ),
            # vm.Card(
            #     text="""
            #         ![🛠️ Advanced Analytics & Machine Learning at Scale ⛅️](https://analytics-experience.pages.dev/assets/images/Advanced-Analytics-Machine-Learning-at-Scale-a64ec9faad408f3deb86225de7b3fbee.gif)
            #     """
            # ),
        ],
    )

    tab_3 = vm.Container(
        title="🌏 Auckland Air Quality Dataset",
        # layout=vm.Layout(grid=[[0, 1], [2, 3]], row_gap="18px", col_gap="18px"),
        # layout=vm.Layout(grid=[[0,1]], row_min_height="500px"),
        layout=vm.Layout(grid=[[0,0,1]]),
        components=[
            vm.Card(
                text="""
                    ![](assets/images/Auckland-Maps.gif#auckland-maps)
                """,
                href="https://analytics-experience.pages.dev/html/Air-Quality-Monitoring-Network",
            ),
            vm.Card(
                text="""
                    Dataset
                    """,
                href="/feature-correlation-visualization",
            ),
            # vm.Card(
            #     text="""
            #         ### 💹 

            #         > 
                    
            #         ...
            #     """,
            #     href="/executive-summary",
            # ),
            # vm.Card(
            #     text="""
            #         ### 🧠 **RQ2.** Predictive Analytics Models/Algorithms

            #         > **RQ2:** What _time series_, machine learning and deep learning models offer the best performance to predict PM2.5/PM10 levels based on data from the Environmental Data in Auckland?
                    
            #         &nbsp;
                    
            #         🌟 Comparative Benchmark Analysis: Discovering how the metrics differ for each location compared to WHO air quality guidelines and export data for further investigation.
            #     """,
            #     href="/predictive-analytics-models-and-algorithms",
            # ),
        ],
    )



    # ##Create a customized AgGrid display with conditional formatting based on environmental monitoring data.
    # ## Define custom cell style for 'Site Name' based on conditions
    # cellStyle = {
    #     "styleConditions": [
    #         {
    #             "condition": "params.node.data['Is Meteorological'] === 'NO'",
    #             "style": {"color": "blue"},
    #         },
    #         {
    #             "condition": "params.node.data['PM25 Monitored'] === 'NO'",
    #             "style": {"color": "green"},
    #         },
    #         {
    #             "condition": "params.node.data['Is Meteorological'] === 'YES' && params.node.data['PM25 Monitored'] === 'YES'",
    #             "style": {"color": "orange"},
    #         },
    #     ]
    # }
    # ## Define column properties, highlight 'Site Name'
    # columnDefs = [
    #     {"field": "Site ID", "cellStyle": cellStyle},
    #     {"field": "Site Name", "cellStyle": cellStyle},
    #     # {"field": "Country"},
    #     # {"field": "City"},
    #     {"field": "Pollutants Monitored", "cellStyle": cellStyle},
    #     # {"field": "PM2.5 Monitored"},
    #     {"field": "Meteorological Measured", "cellStyle": cellStyle},
    #     # {"field": "Is Meteorological"},
    #     {"field": "Site Class", "cellStyle": cellStyle},
    #     # {"field": "latitude"},
    #     # {"field": "longitude"},
    #     # {"field": "Site Address"},
    #     {"field": "Established Date", "cellStyle": cellStyle},
    # ]
    
    # tab_2 = vm.Container(
    #     title="📂 Environmental Dataset",
    #     layout=vm.Layout(grid=[[0],[1]], row_min_height="550px"),
    #     # layout=vm.Layout(grid=[[0,0,1],[2,2,2]], row_min_height="450px"),
    #     components=[
    #         vm.Graph(
    #             id="variable_map",
    #             figure=variable_map(
    #                 data_frame=aap_metadata_df,
    #                 location_col="ISO3",
    #                 # location_col="City",
    #                 color_col="PM2.5 Concentration (µg/m³)",
    #                 hover_col="Country Name",
    #                 # hover_col="City",
    #                 frame_col="Measurement Year",
    #                 title="WHO Air Quality Trends by Country Over Time",
    #                 scope="world"  ## Adjust 'scope' as needed ('europe', 'asia', 'north america', etc.)
    #             ),
    #         ),
    #         vm.AgGrid(figure=dash_ag_grid(data_frame=monitoring_sites_metadata_df, 
    #                                       columnSize="sizeToFit",
    #                                       columnDefs=columnDefs)),
    #     ],
    #     # controls=[
    #     #     vm.Parameter(
    #     #         targets=["variable_map.color"],
    #     #         selector=vm.RadioItems(options=["PM2.5 Concentration (µg/m³)", "PM10 Concentration (µg/m³)"], title="Select variable"),
    #     #     )
    #     # ],
    # )
    
    # tab_3 = vm.Container(
    #     title="⚙️ Agile CRISP-DM",
    #     components=[
    #         vm.Card(
    #             text="""
    #                 ![Agile-implementation-of-CRISP-DM](https://analytics-experience.pages.dev/assets/images/Agile-implementation-of-CRISP-DM-5711e3224a2bea3c5bc33b0bafbb5b43.gif)
                    
    #                 🗓️ Project Timeline/Schedule & Deliverables 🚀: https://analytics-experience-calendar.onrender.com 
    #             """
    #         ),
    #     ],
    # )

    # tab_4 = vm.Container(
    #     title="🛠️ Enterprise at Scale",
    #     components=[
    #         vm.Card(
    #             text="""
    #                 ![🛠️ Advanced Analytics & Machine Learning at Scale ⛅️](https://analytics-experience.pages.dev/assets/images/Advanced-Analytics-Machine-Learning-at-Scale-a64ec9faad408f3deb86225de7b3fbee.gif)
    #             """
    #         ),
    #     ],
    # )

    # tab_5 = vm.Container(
    #     title="🌏 Documentation",
    #     components=[
    #         vm.Card(
    #             text="""
    #                 # 🛠️ Advanced Analytics & Machine Learning at Scale ⛅️

    #                 ## 🔎 Research Project: Predicting Air Particulate Matter at Scale

    #                 ---
                    
    #                 ### 🗓️ Project Timeline/Schedule & Deliverables 🚀
                    
    #                 Please refer to https://analytics-experience-calendar.onrender.com
                    
    #                 ### 🛠️ An Agile implementation of CRISP-DM
                    
    #                 Please refer to https://analytics-experience.pages.dev/docs/data-science/project-roadmap
                    
    #                 ### ⛅️ Analytics & Machine Learning at Scale
                    
    #                 Please refer to https://analytics-experience.pages.dev/docs/data-science/project-proposal

    #                 ---

    #                 > ✍️ I would like to express my gratitude to **Dr. Nuttanan Wichitaksorn** as my professor and supervisor, **Dr. Victor Miranda** as STAT995 course leader and coordinator at Auckland University of Technology (**AUT**), **Jason Sharpe** as my supervisor, and also to **Douglas H. Ebel** and **Bokareva Tatiana** from **Teradata®** for their great support 🙏.
    #             """
    #         ),
    #     ],
    # )

    ### BACKUP: markdown &nbsp;
    # ![](assets/images/icons/line-chart.svg#icon-top)
    # ![](assets/images/icons/hypotheses.svg#icon-top)
    # ![](assets/images/icons/collections.svg#icon-top)
    # ![](assets/images/icons/features.svg#icon-top)
    # * [🛠️ An Agile implementation of CRISP-DM](https://analytics-experience.pages.dev/docs/data-science/project-roadmap){:target="_blank"}  
    # * [⛅️ Advanced Analytics & Machine Learning at Scale](https://analytics-experience.pages.dev/docs/data-science/project-proposal) {:target="_blank"}  
    # ![Agile-implementation-of-CRISP-DM](https://analytics-experience.pages.dev/assets/images/Agile-implementation-of-CRISP-DM-5711e3224a2bea3c5bc33b0bafbb5b43.gif)
    #  🗓️ Project Timeline/Schedule & Deliverables 🚀: https://analytics-experience-calendar.onrender.com 
    #  ![🛠️ Advanced Analytics & Machine Learning at Scale ⛅️](https://analytics-experience.pages.dev/assets/images/Advanced-Analytics-Machine-Learning-at-Scale-a64ec9faad408f3deb86225de7b3fbee.gif)

    page_data_preparation = vm.Page(
        title="Data Preparation",
        description="Data Preprocessing and Data Exploration (EDA)",
        # components=[vm.Tabs(tabs=[tab_1, tab_2, tab_3, tab_4, tab_5])], 
        components=[vm.Tabs(tabs=[tab_1, tab_2, tab_3])], 
                   # controls=[
                   #     # vm.Filter(column='Site', selector=vm.Dropdown(value=['ALL'])),
                   #     vm.Filter(column='Site', selector=vm.Dropdown(value="Penrose", multi=False, title="Select Location")),
                   # ],
        )

    return page_data_preparation


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [Data to Viz] 3.1. Variable Analysis

# In[10]:


# def ensure_timestamp_index(df, timestamp_col='Timestamp'):
#     """
#     Ensure the DataFrame uses the timestamp column as a datetime index.
#     Convert a timestamp_col column to datetime and set it as index.
#     """
#     if timestamp_col not in df.columns:
#         raise ValueError(f"Column {timestamp_col} not found in DataFrame.")
#     if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
#         df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
#         # df.set_index(timestamp_col, inplace=True, drop=True)
#         df.set_index('Timestamp', inplace=True, drop=False)  ## Keep the column for later operations

#     # ## Retains the timestamp_col column in the DataFrame after setting it as an index.
#     # if df.index.name != timestamp_col:
#     #     df.set_index(timestamp_col, inplace=True, drop=False)
#     #     # df.set_index(timestamp_col, inplace=True)
    

# def reset_timestamp_index(df, timestamp_col='Timestamp'):
#     """Reset DateTime Index to bring Timestamp back as a column for compatibility."""
#     if df.index.name == timestamp_col:
#         df.reset_index(inplace=True)
        
#         # df.reset_index(inplace=True, drop=True)  ## Reset the index and drop the original index column
#         # df.reset_index(inplace=True, drop=False) ## Ensure the index is added back as a column
#     # if df.index.name != timestamp_col:
#     #     raise ValueError(f"Index '{timestamp_col}' does not match the current DataFrame index.")
#     # df.reset_index(inplace=True)

def resample_data(df, timestamp_col, target_col, sample_interval='M'):
    """
    Resample/Averages time series data based on the specified interval 1M/6M/YTD/1Y/ALL. 
    Aggregate by Day/Month and Plot Daily/Monthly averages.

    When resampling data, setting the timestamp as the index is crucial.
    Also, after resampling, reset the index afterward for ease of further analysis.
    """
    if timestamp_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"🪲 One or both specified columns: '{timestamp_col}' or '{target_col}' are not in the dataframe")
    if not df.index.is_all_dates:
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    ## Ensure the column for dates is in datetime format and set as index for resampling
    # if not pd.api.types.is_datetime64_any_dtype(df.index):
    #     df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    #     df.set_index(timestamp_col, inplace=True)
    # ensure_timestamp_index(df, timestamp_col)

    ## Only set index if necessary
    # if df.index.name != 'Timestamp':
    #     ensure_datetime_index(df, 'Timestamp')
    ## Resample data based on the provided frequency directly
    df.set_index('Timestamp', inplace=True, drop=False)  ## Keep the column for later operations
    resampled_df = df[target_col].resample(sample_interval).mean()
    
    # resampled_df = df[target_col].resample(sample_interval).mean().reset_index()
    # reset_timestamp_index(resampled_df)
    
    return resampled_df

@capture("graph")
def line_plot_time_series_with_slider(data_frame: pd.DataFrame, site_name='Penrose', timestamp_col='Timestamp', y_column='PM2.5', title='Average/Mean Concentration Over Time', freq='H', height=900):
    """
    Line plots dynamically analyse data over hours, days, months, and years.
    
    Plot time series data after resampling to a given frequency.
    Line Plot for time series data with a Range Slider and Timestamp Selector for dynamic data exploration.
    Plot time series data with dynamic threshold lines based on WHO 2021 guidelines.
    
    Parameters:
        data_frame (pd.DataFrame): DataFrame containing the time series data.
        site_name (str): Name of the site or category of data.
        timestamp_col (str): The name of the datetime column.
        y_column (str): The name of the column to plot.
        title (str): Title of the plot.
        freq (str): Frequency code for resampling ('H', 'D', 'M', etc.); default is '1M' for monthly.
        
    Raises:
        ValueError: If required columns are not present or datetime conversion fails.
    """
    ## Dynamic thresholds based on the pollutant and frequency
    thresholds = {
        'PM2.5': {'D': 15, 'Y': 5},  ## 1D: Daily/24h and 1Y: Annual limits
        'PM10': {'D': 45, 'Y': 15},  ## 1D: Daily/24h and 1Y: Annual limits
        'NO2': {'D': 25, 'Y': 10},   ## 1D: Daily/24h and 1Y: Annual limits
        ## Add additional pollutants and thresholds as needed
        ## TODO: Daily --> a 99th percentile (i.e. 3–4 exceedance days per year).
    }
    
    if timestamp_col not in data_frame.columns:
        raise ValueError(f"🪲 {timestamp_col} columns are not in the DataFrame!")
    else:
        if y_column not in data_frame.columns:
            raise ValueError(f"🪲 {y_column} columns are not in the DataFrame!")
    
    # ensure_timestamp_index(data_frame, timestamp_col)  ## Convert column to datetime and set as index
    # df_resampled = data_frame[y_column].resample(freq).mean()  ## Resample data based on the provided frequency

    # ensure_timestamp_index(data_frame, 'Timestamp')
    # df_resampled = resample_data(data_frame, y_column, freq)
    # reset_timestamp_index(df_resampled, 'Timestamp')

    ## Only set index if necessary
    # if data_frame.index.name != 'Timestamp':
    #     ensure_timestamp_index(data_frame, 'Timestamp')
    ## Resample data based on the provided frequency directly
    data_frame.set_index('Timestamp', inplace=True, drop=False)  ## Keep the column for later operations
    df_resampled = data_frame[y_column].resample(freq).mean()
    
    ## Plotly automatically uses the index for the x-axis and the Series values for the y-axis when the input is a Series.
    fig = px.line(df_resampled, labels={'value': f'{y_column} (µg/m³)', 'index': timestamp_col},
                  title=f'[{site_name}] {y_column} {title}')

    ## Add guideline thresholds if applicable
    # pollutant_thresholds = thresholds.get(y_column, {})
    # limit = pollutant_thresholds.get(freq)
    # if limit:
    #     fig.add_hline(y=limit, line_dash="dot",
    #                   annotation_text=f"2021 WHO Guideline: {limit} µg/m³",
    #                   annotation_position="bottom right")

    # guideline = thresholds.get(y_column, {}).get(freq[1].lower(), None)
    # if guideline:
    #     fig.add_hline(y=guideline, line_dash="dot", annotation_text=f"2021 WHO Guideline: {guideline} µg/m³", annotation_position="bottom right")

    ## Adding guideline thresholds if applicable
    # pollutant_thresholds = thresholds.get(y_column, {})
    # limit = pollutant_thresholds.get(freq)
    # if limit:
    #     fig.add_hline(y=limit, line_dash="dot", line_color='red',
    #                   annotation_text=f"WHO 2021 Guideline (Average of 24 Hours): {limit} µg/m³",
    #                   annotation_position="top right")
    ## Add dynamic guideline thresholds based on the resampling frequency
    # limit = thresholds.get(y_column, {}).get(freq)
    limit = thresholds.get(y_column, {}).get('D')
    if limit is not None:
        fig.add_hline(y=limit, line_dash="dot", line_color='red',
                      # annotation_text=f"2021 WHO Guideline: {limit} µg/m³ ({freq}) ",
                      annotation_text=f"2021 WHO Guideline: {limit} µg/m³ (Average of 24 Hours)",
                      annotation_position="bottom right")
    
    ## Configuring the range slider and buttons for data navigation
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                {"step": "all", "label": "All"},
                {"count": 1, "label": "1 Year", "step": "year", "stepmode": "backward"},
                {"count": 1, "label": "YTD", "step": "year", "stepmode": "todate"},
                {"count": 6, "label": "6 Months", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1 Month", "step": "month", "stepmode": "backward"},
                {"count": 7, "label": "1 Week", "step": "day", "stepmode": "backward"},
                {"count": 1, "label": "1 Day", "step": "day", "stepmode": "backward"},
            ]
        )
    )
    
    fig.update_layout(height=height, 
                      # template='plotly_dark',  ## Apply dark theme for better visibility
                      showlegend=False)
    # fig.show()
    return fig


@capture("graph")
def scatter_plot_time_series_by_hour(data_frame: pd.DataFrame, site_name='Penrose', timestamp_col='Timestamp', y_column='PM2.5', title='Average/Mean Concentration by Hour of Day', height=600):
    """
    Generates an interactive scatter plot showing data points for each hour of each day.
    """
    if timestamp_col not in data_frame.columns:
        raise ValueError(f"{timestamp_col} column are not in the DataFrame!")
    else:
        if y_column not in data_frame.columns:
            raise ValueError(f"{y_column} column are not in the DataFrame!")

    data_frame[timestamp_col] = pd.to_datetime(data_frame[timestamp_col], errors='coerce')
    data_frame['Hour'] = data_frame[timestamp_col].dt.hour  # Extract hour from datetime
    # data_frame['Day'] = data_frame[timestamp_col].dt.date   # Extract date for grouping

    scatter = [
        go.Scatter(
            x=data_frame['Hour'],
            y=data_frame[y_column],
            mode='markers',
            marker=dict(
                color=data_frame[y_column],
                showscale=True,
                colorscale='Rainbow',  ## Adjusted color scale for better visibility: 'OrRd' | 'Viridis' | 'Rainbow' | Bluered
                colorbar=dict(title=y_column),
                size=9,
                opacity=0.55
            )
        )
    ]

    layout = go.Layout(
        title=f'[{site_name}] {y_column} {title}',
        # xaxis=dict(title='Hour of Day'),
        xaxis=dict(
            title='Hour of Day',
            tickmode='array',
            # tickvals=list(range(24)),      ## Ensuring every hour is marked
            tickvals=list(range(0, 24, 2)),  ## Hours from 0 to 23 every 2 hours
            # ticktext=[f"{h}:00" for h in range(24)]  ## Label hours as '0:00', '1:00', ..., '23:00'
            ticktext=[f'{h:02}:00' for h in range(0, 24, 2)]  ## Format labels as 'HH:00'
        ),
        yaxis=dict(title=f'{y_column} (µg/m³)'),
        height=height,
        # template='plotly_dark'
    )

    fig = go.Figure(data=scatter, layout=layout)
    # fig.show()
    return fig

## Function to visualize hourly trends per day of the week
@capture("graph")
def scatter_pivot_hourly_trends_by_day(data_frame: pd.DataFrame, site_name='Penrose', timestamp_col='Timestamp', y_column='PM2.5', title='Hourly Trends per Day of the Week', height=450):
    """
    Visualize hourly trends per day of the week.
    
    Parameters:
        data_frame (pd.DataFrame): DataFrame containing the data to plot.
        site_name (str): Name of the site to include in the title.
        timestamp_col (str): Column name for the timestamp data.
        y_column (str): Column name of the data to plot.
        title (str): Title of the plot.
    """
    if timestamp_col not in data_frame.columns or y_column not in data_frame.columns:
        raise ValueError(f"Missing required columns in DataFrame: {timestamp_col} or {y_column}")

    ## Ensure the Timestamp column is in datetime format and set it as index if not already
    # ensure_timestamp_index(data_frame, timestamp_col)
    data_frame.set_index('Timestamp', inplace=True, drop=False)  ## Keep the column for later operations
    

    ## Debug: Check the preparation of necessary columns
    if 'DayOfWeekName' not in data_frame.columns:
        ## Prepare data by extracting day name of the week: Return Monday..Sunday instead of 0..6
        data_frame['DayOfWeekName'] = data_frame[timestamp_col].dt.day_name()

    ## Generate a scatter plot trace for each day of the week with predefined colors
    ## Light red for Monday, Lighter red for Tuesday, Salmon for Wednesday, Dark salmon for Thursday, Tomato for Friday, Orange red for Saturday, Red for Sunday
    # colors = ['#ffcccb', '#fabebe', '#fa8072', '#e9967a', '#ff6347', '#ff4500', '#ff0000']
    # colors = ['#fabebe', '#fa8072', '#e9967a', '#ff6347', '#ff4500', '#ff0000', '#ffcccb']  ## Color for each day
    colors = [
        '#FF6347',  ## Tomato - Monday
        '#4682B4',  ## Steel Blue - Tuesday
        '#32CD32',  ## Lime Green - Wednesday
        '#FFD700',  ## Gold - Thursday
        '#6A5ACD',  ## Slate Blue - Friday
        '#FF4500',  ## Orange Red - Saturday
        '#20B2AA'   ## Light Sea Green - Sunday
    ]

    ## Reorder Days of the Week to ensure plots follow the usual weekly order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    ## Pivot Table to get average y_column values for each hour across each day of the week
    data_frame_pivot = data_frame.pivot_table(index='Hour', columns='DayOfWeekName', values=y_column, aggfunc='mean')
    data_frame_pivot = data_frame_pivot[day_order]  ## Reorder columns based on day_order

    ## Fill any NaN values which might occur due to no data points at specific hours or days
    data_frame_pivot.fillna(0, inplace=True)
    
    ## Generate plot traces for each day with predefined colors
    traces = [go.Scatter(x=data_frame_pivot.index, y=data_frame_pivot[day], mode='lines+markers', name=day, line=dict(color=color))
              for day, color in zip(day_order, colors)]

    ## Plot layout settings
    layout = go.Layout(
        title={'text': f'[{site_name}] {y_column} {title}', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis={'title': 'Hour of Day'},
        yaxis={'title': y_column},
        height=height,
        # template='plotly_dark'
    )

    ## Create and display the figure
    fig = go.Figure(data=traces, layout=layout)
    # iplot(fig)
    return fig

@capture("graph")
def enhanced_seasonal_box_plot(data_frame, site_name='Penrose', season_col='Season', timestamp_col='Hour', y_column='PM2.5', title='Distribution by the Hour in each Season', xtitle='Hour of Day', ytitle='Level (µg/m³)', height=600):
    """
    Generate an enhanced box plot visualizing PM2.5/PM10 levels across different hours of the day for each season using dynamic labeling and colors..

    Parameters:
        data_frame (pd.DataFrame): DataFrame containing the environmental data.
        season_col (str): Name of the column in data_frame specifying the season.
        timestamp_col (str): Name of the column in data_frame specifying the hour of the day.
        y_column (str): Name of the column in data_frame specifying the target feature/variable.
        title (str): Title for the plot; defaults to a generic title if None.

    Returns:
        None; displays the plot directly.
    """
    ## Map season codes to names
    season_names = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}
    if 'SeasonName' not in data_frame.columns:
        data_frame['SeasonName'] = data_frame[season_col].map(season_names)

    fig = px.box(data_frame, x=timestamp_col, y=y_column, color='SeasonName',
                 category_orders={season_col: list(season_names.values())},
                 title=f'[{site_name}] {y_column} {title}', 
                 labels={timestamp_col: xtitle, y_column: ytitle})

    fig.update_layout( 
        title={'x': 0.5, 'xanchor': 'center'},
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=10,
            title=xtitle,            ## Placeholder, replace with actual x-axis title
            titlefont=dict(size=14)  ## Corrected key from 'titlefont' to 'title_font'
        ), 
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=10,
            title=f'{y_column} {ytitle}',
            titlefont=dict(size=14)
        ), 
        height=height
    )

    # fig.show()
    return fig

@capture("graph")
def enhanced_bar_plot(data_frame, site_name='Penrose', group_by_col='Month', y_column='PM2.5', aggregation_func='sum', plot_title='Concentration per Season (Season Aggregation)', x_axis_title='Month', y_axis_title="(µg/m³)", color_scale='OrRd', height=450):
    """
    Create a bar plot aggregating data by a specified aggregation_func function.
    With enhanced features like dynamic coloring and outlier highlighting.
    
    Parameters:
        data_frame (pd.DataFrame): DataFrame containing the data to plot.
        group_by_col (str): Column name to group by.
        y_column (str): Column name to aggregate.
        aggregation_func (function): Aggregation function (np.sum, np.mean, etc.) to apply.
        plot_title (str): Title of the plot.
        x_axis_title (str): Title for the x-axis.
        y_axis_title (str): Title for the y-axis.
        color_scale (list or str): Color scale for bars.
        height (int): Height of the plot in pixels.
    """
    ## Validate column existence
    if group_by_col not in data_frame.columns or y_column not in data_frame.columns:
        raise ValueError("One or both specified columns are not in the DataFrame.")

    ## Aggregate the data
    aggregated_data = data_frame.groupby(group_by_col).agg({y_column: aggregation_func}).reset_index()

    ## Ensure proper formatting of the x-axis values
    if group_by_col == 'Year':
        aggregated_data[group_by_col] = aggregated_data[group_by_col].astype(str)
    elif group_by_col == 'Month':  ## Formatting for Month names if applicable
        aggregated_data[group_by_col] = pd.to_datetime(aggregated_data[group_by_col], format='%m').dt.month_name()

    ## Create the bar trace
    trace = go.Bar(
        x=aggregated_data[group_by_col],
        y=aggregated_data[y_column],
        text=aggregated_data[y_column].apply(lambda x: f'{x:,.2f}'),
        marker=dict(color=aggregated_data[y_column], colorscale=color_scale)  ## Applying color scale
    )

    ## Set up the layout
    layout = go.Layout(
        title={'text': f'[{site_name}] Total {y_column} {plot_title}', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis={'title': x_axis_title},
        # yaxis={'title': y_axis_title, 'tickformat': ',.2f'},
        yaxis=dict(title=f"{y_column} {y_axis_title}", tickformat=',.2f'),
        height=height,
        margin=dict(l=50, r=50, t=70, b=30),  ## Adjust margins to ensure no cutoff
        # template='plotly_dark',
    )
    
    ## Create and display the figure
    fig = go.Figure(data=[trace], layout=layout)
    
    ## Add median line using horizontal line
    q1 = aggregated_data[y_column].quantile(0.25)
    # median_value = aggregated_data[y_column].quantile(0.5)
    median_value = aggregated_data[y_column].median()
    q3 = aggregated_data[y_column].quantile(0.75)
    fig.add_hline(y=median_value, line_dash="dot",
                  annotation_text="Median", annotation_position="top right")
    ## Add quartile lines
    fig.add_hline(y=q1, line_dash="dot", line_color="blue",
                  annotation_text="Q1 (Quantile 25%)", annotation_position="top left")
    # fig.add_hline(y=median_value, line_dash="dot", line_color="green",
    #               annotation_text="Median", annotation_position="top left")
    fig.add_hline(y=q3, line_dash="dot", line_color="red",
                  annotation_text="Q3 (Quantile 75%)", annotation_position="top left")
    
    # iplot(fig)
    return fig


# In[11]:


def create_variable_analysis(df, site_name='Penrose', timestamp_col='Timestamp', target_col='PM2.5', feature1_col='NO2', feature2_col='Wind_Speed'):
    """Function returns a page with data to do variable analysis."""
    target_col_name = target_col.replace('.', '')

    page_variable = vm.Page(
        title=f"[{target_col_name}] Variable Analysis for {site_name}",
        description="Univariate Analysis of Continuous Variables, including both target and features variables.",
        layout=vm.Layout(
            grid=[
                [0, 1, 1, 1],  ## Card (EDA.1), Graph spans across three columns
                [2, 3, 3, 3],  ## Card (EDA.2), Graph spans across three columns
                [4, 5, 5, 5],  ## Card (EDA.3), Graph spans across three columns
                [6, 7, 7, 7],  ## Card (EDA.4), potentially a Graph spanning three columns
                [8, 9, 9, 9],  ## Card (EDA.5), potentially a Graph spanning three columns
                [10, 11, 11, 11],  ## Card (EDA.5), potentially a Graph spanning three columns
            ],
            row_min_height="400px",
            row_gap="24px",
        ),
        components=[
            vm.Card(
                text=f"\n📈 [EDA.1][Line Plot][Time-Series Plot] Average/Mean Hourly Visualizing {target_col} for {site_name}: Line Plot with a Range Slider and Timestamp Selector over hours, days, months, years."
            ),
            vm.Graph(
                id=f"variable_plot_time_series_with_slider_{site_name}_{target_col_name}",
                ## print(f"\n📈 [EDA.1.1][Line Plot]] Average/Mean Hourly Visualizing {target_col} for {site_name}: Line Plot with a Range Slider and Timestamp Selector over hours, days, months, years.")
                figure=line_plot_time_series_with_slider(data_frame=df, site_name=site_name, timestamp_col='Timestamp', y_column=target_col, title='Average/Mean Concentration Over Time', freq='H')
            ),
            vm.Card(
                text=f"\n📈 [EDA.2][Scatter Plot][Hours of the Day] Average/Mean Hourly Visualizing {target_col} for {site_name}: Scatter Plot Time Series by Hour of Day."
            ),
            vm.Graph(
                id=f"variable_scatter_plot_time_series_by_hour_{site_name}_{target_col_name}",
                figure=scatter_plot_time_series_by_hour(data_frame=df, site_name=site_name, timestamp_col='Timestamp', y_column=target_col, title='Average/Mean Concentration by Hour of Day')
            ),
            vm.Card(
                text=f"\n📈🎓 [EDA.3: ][Pivot Table][Hour of the Day per Every Day of the Week] Hourly Trends per Every Day of the Week {target_col} for {site_name}"
            ),
            vm.Graph(
                id=f"variable_scatter_pivot_hourly_trends_by_day_{site_name}_{target_col_name}",
                figure=scatter_pivot_hourly_trends_by_day(data_frame=df, site_name=site_name, timestamp_col='Timestamp', y_column=target_col, title='Hourly Trends per Day of the Week')
            ),
            vm.Card(
                text=f"\n📈🎓 [EDA.4: ][Box Plot][Hour of the Day in each Season] {target_col} for {site_name}: Distribution by the Hour in each Season"
            ),
            vm.Graph(
                id=f"variable_enhanced_seasonal_box_plot_{site_name}_{target_col_name}",
                figure=enhanced_seasonal_box_plot(data_frame=df, site_name=site_name, season_col='Season', timestamp_col='Hour', y_column=target_col, title=f'Distribution by the Hour in each Season', xtitle='Hour of Day', ytitle=f'{target_col} Level (µg/m³)')
            ),
            vm.Card(
                text=f"\n📈🎓 [EDA.5: ][Bar Plot][Monthly Aggregation: Total/Sum of PMx Concentration per Month] Total Monthly Visualizing {target_col} for {site_name}: [Bar Plot] Total Concentration per Month.\n"
            ),
            vm.Graph(
                id=f"variable_enhanced_bar_plot_month_{site_name}_{target_col_name}",
                figure=enhanced_bar_plot(data_frame=df, site_name=site_name, group_by_col='Month', y_column=target_col, aggregation_func='sum', plot_title='Concentration per Month (Monthly Aggregation)', x_axis_title='Month', y_axis_title="(µg/m³)")
            ),
            vm.Card(
                text=f"\n📈🎓 [EDA.5: ][Bar Plot][Seasonal Aggregation: Total/Sum of PMx Concentration per Season] Total Seasonal Visualizing {target_col} for {site_name}: [Bar Plot] Total Concentration per Season.\n"
            ),
            vm.Graph(
                id=f"variable_enhanced_bar_plot_season_{site_name}_{target_col_name}",
                figure=enhanced_bar_plot(data_frame=df, site_name=site_name, group_by_col='Season', y_column=target_col, aggregation_func='sum', plot_title='Concentration per Season (Season Aggregation)', x_axis_title='Season', y_axis_title="(µg/m³)")
            ),
        ],
        # controls=[
        #     vm.Parameter(
        #         # targets=["variable_map.color", "variable_boxplot.y", "variable_line.y", "variable_bar.x"],
        #         selector=vm.RadioItems(options=[target_col, feature1_col, feature2_col], title="Select Variable"),
        #     )
        # ],
    )
    return page_variable


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [Data to Viz] 3.2. Feature Correlation (Sankey Diagram)

# <div class="alert alert-block alert-info">
# 🎓 In the section, Plot the data and try to extract some knowledge.</p>
# 
# * [x] **Sankey Diagram**: a three-column structure in the Sankey diagram clearly delineates the flow from specific features of one location through the common target pollutants to the features of another location. This visualization strategy enhances the understanding of how features in different locations (Penrose and Takapuna) relate to the same environmental pollutants (PM10 and PM2.5). This setup not only offers a visual comparison between the two locations but also emphasizes the intermediary role of the pollutants, thereby highlighting potential areas for targeted environmental control.
# * [ ] **dash_ag_grid Grid**: comparative analysis across multiple variables and sites, the correlation between the row (feature) and the columns (site and pollutant) needs to be properly highlighted and utilized. Each column definition now includes conditional formatting for cell color based on the site (Penrose or Takapuna) and cell background color based on correlation strength.
#   * {"backgroundColor": "#4CAF50", "color": "white"}  ## Strong positive correlation
#   * {"backgroundColor": "#FFEB3B", "color": "black"}  # Moderate positive correlation
#   * {"backgroundColor": "#F44336", "color": "white"}  # Weak correlation
# 
# </div>

# In[12]:


def features_to_exclude(target):
    """ Returns a list of features to exclude, including target and its lag variables. """
    return [target, f"{target}_Lag1", f"{target}_Lag2"]

def prepare_sankey_data(top_features_data11, top_features_data12, top_features_data21, top_features_data22):
    ## Extract the lists of tuples to easily handle the logic
    pm25_exclusions = features_to_exclude('PM2.5')
    pm10_exclusions = features_to_exclude('PM10')

    ## Initialize records for DataFrame
    records = []

    ## Filtering and adding links from Penrose PM2.5/PM10 to feature variables
    ## abs(value) --> round(abs(value) * 100, 2)  ## Convert to percentage and round off
    for dataset, source in [(top_features_data11, 'Penrose PM2.5')]:
        for feature, value in dataset:
            if feature not in pm25_exclusions:
                records.append({'Source': source, 
                                'Target': f'Feature - {feature}', 
                                'Value': round(abs(value) * 100, 1), })

    ## Filtering and adding links from Penrose PM2.5/PM10 to feature variables
    for dataset, source in [(top_features_data21, 'Penrose PM10')]:
        for feature, value in dataset:
            if feature not in pm10_exclusions:
                records.append({'Source': source, 
                                'Target': f'Feature - {feature}', 
                                'Value': round(abs(value) * 100, 1)})

    ## Filtering and adding links from feature variables to Takapuna PM2.5/PM10
    for dataset, target in [(top_features_data12, 'Takapuna PM2.5')]:
        for feature, value in dataset:
            if feature not in pm25_exclusions:
                records.append({'Source': f'Feature - {feature}', 
                                'Target': target, 
                                'Value': round(abs(value) * 100, 1)})

    ## Filtering and adding links from feature variables to Takapuna PM2.5/PM10
    for dataset, target in [(top_features_data22, 'Takapuna PM10')]:
        for feature, value in dataset:
            if feature not in pm10_exclusions:
                records.append({'Source': f'Feature - {feature}', 
                                'Target': target, 
                                'Value': round(abs(value) * 100, 1)})

    return pd.DataFrame(records)


def assign_link_colors(data_frame, source_color_map, default_color):
    ## Initialize a list to store colors for each link
    link_colors = []
    
    ## Iterate through each row in the data frame to determine the appropriate color
    for idx, row in data_frame.iterrows():
        source = row['Source']
        target = row['Target']
        
        ## Default to using the source for color mapping
        color = source_color_map.get(source, None)
        
        ## If the source is not in the map, try the target
        if color is None:
            color = source_color_map.get(target, default_color)
        
        ## Append the determined color to the list
        link_colors.append(color)
    
    return link_colors

@capture("graph")
def create_sankey_diagram(data_frame: pd.DataFrame, title="PM2.5/PM10 Feature Flow in Penrose and Takapuna"):
    """
    Assign colors based on the type of link or the source/target attributes, enhancing the interpretability of the diagram
    """
    ## Extract and map labels to indices
    labels = pd.concat([data_frame['Source'], data_frame['Target']]).unique()
    label_indices = {label: idx for idx, label in enumerate(labels)}

    ## Initialize color map for sources and default color for features
    # source_color_map = {
    #     'Penrose PM2.5': 'rgba(75, 192, 192, 0.6)', ## Cyan
    #     'Penrose PM10': 'rgba(255, 159, 64, 0.6)',  ## Orange
    #     'Takapuna PM2.5': 'rgba(255, 99, 71, 0.6)', ## Tomato
    #     'Takapuna PM10': 'rgba(54, 162, 235, 0.6)'  ## Blue
    # }
    source_color_map = {
        'Penrose PM2.5': 'rgba(33, 113, 181, 0.6)',  ## Adjusted to a shade of blue
        'Penrose PM10': 'rgba(253, 141, 60, 0.6)',   ## Adjusted to a shade of orange
        # 'Takapuna PM2.5': 'rgba(225, 25, 28, 0.6)',  ## Adjusted to a shade of red
        'Takapuna PM2.5': 'rgba(255, 99, 71, 0.6)',  ## Tomato
        'Takapuna PM10': 'rgba(35, 132, 67, 0.6)'    ## Adjusted to a shade of green
    }
    default_feature_color = 'rgba(123, 104, 238, 0.5)'  ## Semi-transparent purple
    ## Penrose:  link_colors = [source_color_map.get(source, default_feature_color) for source in data_frame['Source']]
    ## Takapuna: link_colors = [source_color_map.get(target, default_feature_color) for target in data_frame['Target']]
    ## Assign colors dynamically to links based on the target for better visual representation
    link_colors = assign_link_colors(data_frame, source_color_map, default_feature_color)

    ## Map source and target text to indices
    data_frame['Source_idx'] = data_frame['Source'].map(label_indices)
    data_frame['Target_idx'] = data_frame['Target'].map(label_indices)

    ## Building the Sankey diagram
    ## node_pad=15, node_thickness=20, link_color="rgba(123, 104, 238, 0.5)"
    fig = go.Figure(data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels
            ),
            link=dict(
                source=data_frame['Source_idx'],
                target=data_frame['Target_idx'],
                value=data_frame['Value'],
                # color="rgba(123, 104, 238, 0.5)"  ## Semi-transparent purple for visibility
                color=link_colors
            )
        )
    ])
    fig.update_layout(title_text=title, font_size=12)
    return fig


# In[64]:


def create_feature_correlation_visualization():

    ## 'PM2.5' Analyzing for both sites
    top_features_data11 = DataFrameAdapter.get_top_correlated_features(data=cleaned_data_site1, target='PM2.5', num_features=10)
    top_features_data12 = DataFrameAdapter.get_top_correlated_features(cleaned_data_site2, 'PM2.5', num_features=10)
    
    # print("\n🌟 top_features_data11: Top 10 features highly correlated with PM2.5 in Penrose: %s\n", top_features_data11)
    # print("\n🌟 top_features_data12: Top 10 features highly correlated with PM2.5 in Takapuna: %s\n", top_features_data12)
    
    ## 'PM10' Analyzing for both sites
    top_features_data21 = DataFrameAdapter.get_top_correlated_features(data=cleaned_data_site1, target='PM10', num_features=10)
    top_features_data22 = DataFrameAdapter.get_top_correlated_features(cleaned_data_site2, 'PM10', num_features=10)
    
    # print("\n🌟 top_features_data21: Top 10 features highly correlated with PM10 in Penrose: %s\n", top_features_data21)
    # print("\n🌟 top_features_data22: Top 10 features highly correlated with PM10 in Takapuna: %s\n", top_features_data22)

    ## Note: The top_features_data for PM2.5 and PM10 from Penrose and Takapuna have been defined
    df_sankey = prepare_sankey_data(top_features_data11, top_features_data12, top_features_data21, top_features_data22)
    
    sankey_fig = create_sankey_diagram(data_frame=df_sankey, title="PM2.5/PM10 Feature Flow in Penrose and Takapuna")
    # sankey_fig.show()
    
    ## Creating a Vizro page to display the Sankey diagram
    page_feature_correlation_visualization = vm.Page(
        title="Feature Correlation Visualization",
        description="Dynamic relationships between Pollutants & Meteorological variables over time & locations.",
        components=[
            vm.Graph(figure=sankey_fig),
        ]
    )
    
    return page_feature_correlation_visualization


# In[65]:


## [DEBUG]

# Vizro._reset()

# ## Creating a Vizro page to display the Sankey diagram
# sankey_page = vm.Page(
#     title="Feature Correlation Visualization",
#     components=[
#         vm.Graph(figure=sankey_fig),
#     ]
# )

# dashboard = vm.Dashboard(pages=[sankey_page])

# Vizro(assets_folder="assets").build(dashboard).run(port=8083)


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [Data to Viz] 3.x. Benchmark Analysis

# In[66]:


def create_benchmark_analysis():
    """Function returns a page to perform analysis on location level."""
    # Apply formatting to grid columns
    cellStyle = {
        "styleConditions": [
            {
                "condition": "params.value < 1045",
                "style": {"backgroundColor": "#ff9222"},
            },
            {
                "condition": "params.value >= 1045 && params.value <= 4095",
                "style": {"backgroundColor": "#de9e75"},
            },
            {
                "condition": "params.value > 4095 && params.value <= 12695",
                "style": {"backgroundColor": "#aaa9ba"},
            },
            {
                "condition": "params.value > 12695",
                "style": {"backgroundColor": "#00b4ff"},
            },
        ]
    }
    columnsDefs = [
        {"field": COL_LOCATION},
        {"field": COL_COUNTRY},
        {"field": COL_TIMESTAMP},
        {"field": COL_TARGET, "cellDataType": "numeric"},
        {"field": COL_FEATURE1, "cellDataType": "dollar", "cellStyle": cellStyle},
        {"field": COL_FEATURE2},
    ]

    page_benchmark_analysis = vm.Page(
        title="Benchmark Analysis",
        description="Discovering how the metrics differ for each location and export data for further investigation",
        layout=vm.Layout(grid=[[0, 1]] * 5 + [[2, -1]], col_gap="32px", row_gap="60px"),
        components=[
            vm.AgGrid(
                title="Click on a cell in location column:",
                figure=dash_ag_grid(id="dash_ag_grid_location", data_frame=df_data_site1, columnDefs=columnsDefs),
                actions=[vm.Action(function=filter_interaction(targets=["line_location"]))],
            ),
            vm.Graph(
                id="line_location",
                figure=px.line(
                    df_data_concat,
                    title="Location vs. Country/WHO",
                    x=COL_TIMESTAMP,
                    y=COL_FEATURE1,
                    color="color",
                    labels={COL_TIMESTAMP: COL_TIMESTAMP, "data": "Data", COL_FEATURE1: "Feature variable 1"},
                    color_discrete_map={COL_LOCATION: "#afe7f9", "COL_LOCATION": "#003875"},
                    markers=True,
                    hover_name=COL_LOCATION,
                ),
            ),
            vm.Button(text="Export data", actions=[vm.Action(function=export_data(targets=["line_location"]))]),
        ],
        controls=[
            vm.Filter(column=COL_LOCATION, selector=vm.Dropdown(value="Penrose", multi=False, title="Select location")),
            vm.Filter(column=COL_TIMESLIDER, selector=vm.RangeSlider(title="Select timeframe", step=1, marks=None)),
            vm.Parameter(
                targets=["line_location.y"],
                selector=vm.Dropdown(
                    options=[COL_TARGET, COL_FEATURE1, COL_FEATURE2], multi=False, value=COL_FEATURE1, title="Choose y-axis"
                ),
            ),
        ],
    )
    return page_benchmark_analysis


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [Data to Viz] 4. Predictive Analytics Models and Algorithms

# In[67]:


def create_predictive_analytics(df, timestamp_col='Timestamp', target_col='PM2.5', feature1_col='NO2'):

    tab_1 = vm.Container(
        title="Dataset",
        components=[
            vm.AgGrid(title="Title - AG Grid", figure=dash_ag_grid(data_frame=df)),
        ],
    )
    
    tab_2 = vm.Container(
        title="Line Chart",
        components=[
            # vm.Graph(
            #     id="scatter_chart",
            #     figure  = px.scatter(df, x=feature1_col, y=target_col, color='Site', title=f'[Line Chart] {timestamp_col} vs. {target_col} Visualization across Locations'),
            # ),
            vm.Card(
                text="ARIMA ..."
            ),
        ],
    )

    tab_3 = vm.Container(
        title="Correlation",
        components=[
            # vm.Graph(
            #             figure=px.scatter(
            #                 df,
            #                 x=feature1_col,
            #                 y=target_col,
            #                 color='Site',
            #                 marginal_y="violin",
            #                 marginal_x="box",
            #                 title="Container - Scatter",
            #             )
            #         ),
            vm.Card(
                text="333"
            ),
        ],
    )


    page_predictive_analytics = vm.Page(
        title="Predictive Analytics Models and Algorithms",
        description="Discovering how the metrics differ for each location and export data for further investigation",
        components=[vm.Tabs(tabs=[tab_1, tab_2, tab_3])], 
                   # controls=[
                   #     # vm.Filter(column='Site', selector=vm.Dropdown(value=['ALL'])),
                   #     vm.Filter(column='Site', selector=vm.Dropdown(value="Penrose", multi=False, title="Select Location")),
                   # ],
        )

    return page_predictive_analytics


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [Data to Viz] 5. Executive Summary

# In[123]:


def create_location_executive_summary():
    """Location Executive Summary: Function returns a page with markdown including images."""
    page_summary = vm.Page(
        title="Executive Summary",
        description="Summarizing the main findings for each location",
        # SIZE_OF_LOCATION = 4        ## Penrose & Takapuna
        # layout=vm.Layout(grid=[[i] for i in range(SIZE_OF_LOCATION)], row_min_height="250px", row_gap="5px"),
        # layout=vm.Layout(grid=[[0, 0], [1, 2], [1, 2], [1, 2]]),
        layout=vm.Layout(grid=[[0, 0], [1, 2], [3, 4]]),
        components=[
            vm.Card(
                text="""
                    Our research aimed to accurately predict air particulate matter (**PM2.5** and **PM10**) using **time series** and **machine learning** methods in two suburbs of New Zealand, Penrose and Takapuna. 
                    
                    We develop the models using `ARIMA`, `Prophet`, `NeuralProphet`, `LinearRegression`, `Ridge`, `Lasso`, `RandomForest`, `SVR`, `XGBoost`, and evaluate them based on `RMSE`, MSE, MAE, R2, and Adjusted R2 Score. The `RandomForest` models may perform best for 3 target variables: PM2.5 and PM10 in Penrose, while `SVR` may perform best for PM10 in Takapuna. 
                    
                    Future research could explore **hybrid models** that combine RandomForest with other algorithms to capture both temporal and complex feature interactions, **integrate traffic and industrial activity data**, and utilise interpretability techniques like SHAP to improve model usefulness for policymakers and stakeholders.
                """
            ),
            vm.Card(
                text="""
                    | ![](assets/images/locations/Penrose.png#my-image) | ![](assets/images/Feature-Importances/Penrose.png#my-image-2) |
                    |:-------------------------------------------------:|:----------------------------------------------------------:|
                """
            ),
            vm.Card(
                text="""
                    | ![](assets/images/locations/Takapuna.png#my-image) | ![](assets/images/Feature-Importances/Takapuna.png#my-image-2) |
                    |:-------------------------------------------------:|:----------------------------------------------------------:|
                """
            ),
            vm.Card(
                text="""
                    ##### PM2.5 & PM10 in Penrose
                
                    * Across both locations, `PM10` and its lag were the most important predictors for `PM2.5`, emphasising temporal dependencies across both locations.
                    * Variables like nitric oxide `NO`, `wind speed`, and `relative humidity` contribute significantly to `PM10` levels in Penrose, indicating the influence of vehicle emissions and weather conditions.
                """
            ),
            vm.Card(
                text="""
                    ##### PM2.5 & PM10 in Takapuna
                    
                    * Air Temperature may contribute significantly to PM2.5 levels at both sites, possibly through changes in atmospheric stability and pollution spread.
                    * Nitrogen oxides `NOx`, `wind speed`, and temporal factors such as the `hour of day` have an important effect in Takapuna, reflecting the impact of traffic emissions and wind patterns.
                """
            ),
        ],
    )
    return page_summary


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [Data to Viz] 🚀 Advanced Analytics Dashboard 🌟

# In[124]:


Vizro._reset()


# In[125]:


dashboard = vm.Dashboard(
    title="Advanced Analytics & Machine Learning at Scale",
    pages=[
        create_home_page(),
        create_data_preparation_page(),
        create_variable_analysis(df=df_data_site1, site_name='Penrose', timestamp_col='Timestamp', target_col='PM2.5', feature1_col='NO2', feature2_col='Wind_Speed'),
        create_variable_analysis(df=df_data_site2, site_name='Takapuna', timestamp_col='Timestamp', target_col='PM2.5', feature1_col='NO2', feature2_col='Wind_Speed'),
        create_variable_analysis(df=df_data_site1, site_name='Penrose', timestamp_col='Timestamp', target_col='PM10', feature1_col='NO2', feature2_col='Wind_Speed'),
        create_variable_analysis(df=df_data_site2, site_name='Takapuna', timestamp_col='Timestamp', target_col='PM10', feature1_col='NO2', feature2_col='Wind_Speed'),        
        # create_relation_analysis2(df=df_data_site1, site_name='Penrose', numerical_columns=include_columns_site1, timestamp_col='Timestamp', target_col='PM2.5', feature1_col='NO2', feature2_col='NO2', timeslider_sample_interval='1D'),
        create_feature_correlation_visualization(),
        # create_benchmark_analysis(df=cleaned_data, timestamp_col='Timestamp', target_col='PM2.5', feature1_col='NO2'),
        create_predictive_analytics(df=df_data_site1, timestamp_col='Timestamp', target_col='PM2.5', feature1_col='NO2'),
        create_location_executive_summary(),
    ],
    navigation=vm.Navigation(
        nav_selector=vm.NavBar(
            items=[
                vm.NavLink(label="Home", pages=["Home"], icon="Home"),
                vm.NavLink(label="Data Preparation", pages=["Data Preparation"], icon="Database"),
                vm.NavLink(
                    label="Analysis",
                    # pages=["Variable Analysis", "Relationship Analysis", "Benchmark Analysis"],
                    pages=["[PM25] Variable Analysis for Penrose", "[PM25] Variable Analysis for Takapuna", "[PM10] Variable Analysis for Penrose", "[PM10] Variable Analysis for Takapuna","Feature Correlation Visualization", "Predictive Analytics Models and Algorithms"],
                    icon="Stacked Bar Chart",
                ),
                vm.NavLink(label="Summary", pages=["Executive Summary"], icon="Globe"),
            ]
        ),
    ),
)


# In[126]:


if not IS_JUPYTERLAB:
    app = Vizro().build(dashboard)
    server = app.dash.server
    
    if __name__ == "__main__":  
        app.run()
else:
    Vizro(assets_folder="assets").build(dashboard).run(port=8088)


# <footer style="padding-bottom:35px; background:#f9f9f9; border-bottom:3px solid #00b2b1">
#     <div style="float:left;margin-top:14px;color:#E37C4D">🎓 Predicting Air Particulate Matter at Scale ⛅️</div>
#     <div style="float:right;">
#         <div style="float:left; margin-top:14px">
#             Auckland University of Technology (AUT) 🧑‍🎓 
#         </div>
#     </div>
# </footer>

# <header style="padding:1px;background:#00b2b1;border-top:5px solid #E37C4D">
# 
# # 📊 Reusable UI/UX Components

# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [UI/UX] Sankey Diagram (WIP)

# In[21]:


def transform_data_for_sankey2(top_features_penrose, top_features_takapuna):
    # Collect all unique features and pollutants
    features_penrose = {feat for feat, _ in top_features_penrose}
    features_takapuna = {feat for feat, _ in top_features_takapuna}
    pollutants = ['PM2.5', 'PM10']  # Common targets

    # Create records for source-target-value tuples
    records = []

    # Penrose features to pollutants
    for feature, value in top_features_penrose:
        for pollutant in pollutants:
            records.append({'Source': f'Penrose - {feature}', 'Target': f'Pollutant - {pollutant}', 'Value': value})

    # Pollutants to Takapuna features
    for feature, value in top_features_takapuna:
        for pollutant in pollutants:
            records.append({'Source': f'Pollutant - {pollutant}', 'Target': f'Takapuna - {feature}', 'Value': value})

    return pd.DataFrame(records)


@capture("graph")
def sankey_diagram2(data_frame, title=None):
    # Unique labels for nodes
    unique_nodes = pd.concat([data_frame['Source'], data_frame['Target']]).unique()
    label_indices = {label: idx for idx, label in enumerate(unique_nodes)}

    # Map source and target to indices
    data_frame['Source_idx'] = data_frame['Source'].map(label_indices)
    data_frame['Target_idx'] = data_frame['Target'].map(label_indices)

    # Build the figure
    fig = go.Figure(data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=unique_nodes
            ),
            link=dict(
                source=data_frame['Source_idx'],
                target=data_frame['Target_idx'],
                value=data_frame['Value']
            )
        )
    ])
    fig.update_layout(title_text=title, font_size=10)
    return fig

def merge_feature_data2(data1, data2):
    """
    Merges two lists of tuples (feature, correlation value) by combining correlation values more appropriately.

    Args:
    data1 (list): List of tuples from the first dataset.
    data2 (list): List of tuples from the second dataset.

    Returns:
    list: A sorted list of tuples with features and their combined correlation values.
    """
    combined = {}
    for feature, value in data1 + data2:
        if feature in combined:
            # Take the maximum of the absolute values to ensure the strongest correlation is represented.
            combined[feature] = max(combined[feature], value, key=abs)
        else:
            combined[feature] = value

    # Convert the dictionary to a sorted list of tuples by absolute correlation value, descending.
    return sorted(combined.items(), key=lambda x: abs(x[1]), reverse=True)



# In[22]:


# ## Assuming top_features_data11, top_features_data12, top_features_data21, top_features_data22 are defined
# top_features_penrose = merge_feature_data2(top_features_data11, top_features_data21)
# top_features_takapuna = merge_feature_data2(top_features_data12, top_features_data22)

# print("\n🌟 Combined Top Features for Penrose: %s\n", top_features_penrose)
# print("\n🌟 Combined Top Features for Takapuna: %s\n", top_features_takapuna)


# df_sankey2 = transform_data_for_sankey2(top_features_penrose, top_features_takapuna)
# sankey_fig2 = sankey_diagram2(df_sankey2, title="Feature Flow from Penrose to Takapuna via Pollutants")


# Vizro._reset()

# ## Creating a Vizro page to display the Sankey diagram
# sankey_page = vm.Page(
#     title="Feature Correlation Visualization 2",
#     components=[
#         # vm.AgGrid(
#         #     figure=dash_ag_grid(data_frame=df_feature_data, columnSize="sizeToFit", columnDefs=columnDefs)
#         # ),
#         vm.Graph(figure=sankey_fig2)
#     ]
# )

# # vm.Page(title="Sankey Diagram Analysis", components=[vm.Graph(figure=sankey_fig)])


# dashboard = vm.Dashboard(pages=[sankey_page])

# Vizro(assets_folder="assets").build(dashboard).run(port=8083)


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [UI/UX] Relation Analysis 2 (WIP)

# In[23]:


@capture("graph")
def scatter_relation(data_frame: pd.DataFrame, site_col='Site', timestamp_col='Timestamp', y='PM2.5', x='NO2', size='Wind_Speed'):
    """
    Generates a dynamic scatter plot with animation visualizing relationships over time with respect to different sites.

    Custom scatter plot  that needs post update calls.
    Plot the relationship between air pollutants and meteorological variables across different locations and over time.

    Parameters:
    - data_frame (pd.DataFrame): The DataFrame containing the environmental data.
    - site_col (str): Column name for site data.
    - timestamp_col (str): Column name for the timestamp.
    - y (str): The target variable column name.
    - x (str): The independent variable column name.
    - size (str): The column to use for bubble sizes.
    
    Returns:
    - plotly.graph_objects.Figure: The interactive scatter plot.
    """
    # color_discrete_map={
    #         "Penrose": "#00b4ff",
    #         "Takapuna": "#ff9222",
    #         # "Location 3": "#3949ab",
    #         # "Location 4": "#ff5267",
    #         # "Location 5": "#08bdba",
    #     }
    color_discrete_map = {site: color for site, color in zip(data_frame[site_col].unique(), px.colors.qualitative.Plotly)}
    
    fig = px.scatter(
        data_frame,
        x=x,
        y=y,
        animation_frame=timestamp_col,
        animation_group=site_col,
        size=size,
        size_max=60, ##FIXME
        color=site_col, ## FIXME
        marginal_y="violin", ## FIXME
        hover_name=site_col,
        marginal_x="box",
        # labels={
        #     COL_FEATURE1: "Feature variable 1",
        #     COL_FEATURE2: "Feature variable 2",
        #     COL_TARGET: "Target variable",
        #     COL_LOCATION: COL_LOCATION,
        # },
        labels={y: "Target Variable", x: "Feature Variable 1", size: "Feature Variable 2"},
        range_y=[data_frame[y].min(), data_frame[y].max()],
        color_discrete_map=color_discrete_map,
    )

    fig.update_layout(
        # title="Relationship over time",
        title=f"Dynamic Relationship: {y} vs {x} Over Time",
        legend={"orientation": "v", "yanchor": "bottom", "y": -0.2, "xanchor": "right", "x": 1}
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    return fig


# In[24]:


def resample_numerical_columns(df, site_name='Penrose', timestamp_col='Timestamp', target_cols=None, timeslider_sample_interval='M'):
    """
    Resample time series data for specified multiple columns based on the specified interval.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to resample.
    - timestamp_col (str): Name of the timestamp column.
    - target_cols (list): List of column names to resample.
    - site_name (str): Site name for filtering or other purposes.
    - sample_interval (str): Resampling frequency, default= '1M'. RangeSlider: 'Year' | 'Month'
    
    Returns:
    - pd.DataFrame: Resampled DataFrame.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column {timestamp_col} not found in DataFrame.")

    if not all(col in df.columns for col in target_cols):
        missing = [col for col in target_cols if col not in df.columns]
        raise ValueError(f"Missing columns: {missing} in DataFrame.")

    # ## Adjust based on whether 'Timestamp' is a column or index
    # was_reset = False
    # if timestamp_col != df.index.name:
    #     reset_timestamp_index(df, timestamp_col)
    #     was_reset = True
    
    ## Ensure 'Timestamp' is the index for resampling
    # ensure_timestamp_index(df, timestamp_col)

    df.set_index('Timestamp', inplace=True, drop=False)  ## Keep the column for later operations
    resampled_df = df[target_cols].resample(timeslider_sample_interval).mean()
    
    ## Adjust the timestamp index to a more readable format
    resampled_df.index = resampled_df.index.to_period(timeslider_sample_interval)  ## Converts to YYYY-MM format
    resampled_df['Site'] = site_name
    
    # resampled_df.reset_index(inplace=True) ## Properly resets index now that 'drop=True' in set_index
    ## Convert index back to column if it was previously reset
    # if was_reset:
    #     reset_timestamp_index(resampled_df, timestamp_col)
    
    return resampled_df


def create_relation_analysis2(df, site_name='Penrose', numerical_columns=include_columns_site1, timestamp_col='Timestamp', target_col='PM2.5', feature1_col='NO2', feature2_col='Wind_Speed', timeslider_sample_interval='1M'):
    """
    Function returns a page to perform relation analysis between pollutants and meteorological variables.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the cleaned environmental data.
    - numerical_columns (list): List of numerical columns to include in the analysis.
    - site_name (str): Site name for identification.
    - timestamp_col (str): Column name containing the timestamp data.
    - target_col (str): Target variable for the scatter plot (y-axis).
    - feature1_col (str): First feature variable (x-axis).
    - feature2_col (str): Second feature variable (bubble size).
    - timeslider_sample_interval (str): Resampling frequency for the timeslider.

    Returns:
    - vizro.Page: Vizro page object.
    """
    ## Define dynamic options based on dataframe columns
    feature_variables = [{'label': col, 'value': col} for col in df.select_dtypes(include=['float64', 'int64']).columns]
    target_variables = [{'label': col, 'value': col} for col in ['PM2.5', 'PM10', 'NO2'] if col in df.columns]

    resampled_df = resample_numerical_columns(df, site_name='Penrose', timestamp_col='Timestamp', target_cols=numerical_columns, timeslider_sample_interval=timeslider_sample_interval)
    
    page_relation_analysis2 = vm.Page(
        title="Relationship Analysis 2",
        description="Dynamic relationships between Pollutants & Meteorological variables over time & locations.",
        layout=vm.Layout(
            grid=[[0, 0, 0, 0, 0]] + [[1, 1, 1, 1, 1]] * 4,
            row_min_height="100px",
            row_gap="24px",
        ),
        components=[
            vm.Card(
                text="Explore the relationships between different pollutants and meteorological variables to understand underlying patterns."
            ),
            vm.Graph(
                id="scatter_relation",
                figure=scatter_relation(data_frame=resampled_df, site_col='Site', timestamp_col='Timestamp', y=target_col, x=feature1_col, size=feature2_col)
            ),
        ],
        controls=[
            vm.Parameter(
                targets=["scatter_relation.y"],
                selector=vm.Dropdown(
                    # options=[COL_TARGET, COL_FEATURE1, COL_FEATURE2], multi=False, value=COL_TARGET, title="Choose Target (y-axis)"
                    options=target_variables, multi=False, value=target_col, title="Choose Target Variables (y-axis)"
                ),
            ),
            vm.Parameter(
                targets=["scatter_relation.x"],
                selector=vm.Dropdown(
                    # options=[COL_TARGET, COL_FEATURE1, COL_FEATURE2], multi=False, value=COL_FEATURE1, title="Choose Features Variables (x-axis)"
                    options=feature_variables, multi=False, value=feature1_col, title="Choose Features Variables (x-axis)"
                ),
            ),
            vm.Parameter(
                targets=["scatter_relation.size"],
                selector=vm.Dropdown(
                    # options=[COL_TARGET, COL_FEATURE1, COL_FEATURE2], multi=False, value=COL_FEATURE2, title="Choose Bubble Size"
                    options=feature_variables, multi=False, value=feature2_col, title="Choose Bubble Size"
                ),
            ),
        ],
    )
    return page_relation_analysis2


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [UI/UX] Carousel

# In[25]:


# Vizro._reset()


# In[26]:


## 1. Create new custom component
class Carousel(vm.VizroBaseModel):
    type: Literal["carousel"] = "carousel"
    items: List[dict] = []  ## List of items each with keys: 'key', 'src', 'text'
    actions: List[vm.Action] = []  ## Actions that can be triggered
    controls: bool = True
    indicators: bool = True

    _set_actions = _action_validator_factory("active_index")  

    def build(self):
        ## Returns a dbc.Carousel with items and controls based on class attributes
        return dbc.Carousel(
            id=self.id,
            items=[{"key": item['key'], "src": item['src'], "caption": item.get('text', '')} for item in self.items],
            # controls=self.controls,     ## Enables navigation controls ?
            # indicators=self.indicators, ## Enables navigation indicators ?
            # style={'height': '300px'}  # Ensures carousel height is maintained at 500px
        )

## 2. Add new components to expected type - here the selector of the parent components
vm.Page.add_type("components", Carousel)

## Define the page with the custom Carousel component
def create_carousel_page(items: List[dict], title: str) -> vm.Page:
    """Generates a page with a carousel displaying images and descriptions/texts."""
    # ## 3. Create custom action
    # @capture("action")
    # def handle_carousel_change(active_index: int) -> str:
    #     """Handles carousel slide change to display corresponding slide number."""
    #     ## Proper function definition and return based on active_index
    #     return f"Slide number: {active_index}"

    carousel = Carousel(
        id="my-carousel-devtest2",
        items=items,
        # actions=[
        #     vm.Action(
        #         function=handle_carousel_change(),
        #         inputs=["my-carousel.active_index"],
        #         outputs=["carousel-text-card.children"],
        #     )
        # ]
    )
    
    # card_text = vm.Card(text=items[0]['text'], id="carousel-text-card") ## Default to first item's text

    ## Define a page with the custom Carousel component
    return vm.Page(
        title=title,
        # layout=vm.Layout(grid=[[i] for i in range(2)], row_min_height="500px"),
        layout=vm.Layout(grid=[[i] for i in range(1)], row_min_height="500px"),
        components=[
            # card_text,
            carousel,
        ],
    )


items_descriptive_statistics = [
    {"key": "1", "src": "assets/images/Descriptive-Statistics/correlations_heatmap_with_regression-1.png", "text": "First Image Description"},
    {"key": "2", "src": "assets/images/Descriptive-Statistics/correlations_heatmap_with_regression-2.png", "text": "Second Image Description"},
    {"key": "3", "src": "assets/images/Descriptive-Statistics/correlation-matrix-heatmap-penrose.png", "text": "[Site1 - Penrose] Correlation Matrix Heatmap"},
    {"key": "4", "src": "assets/images/Descriptive-Statistics/correlation-matrix-heatmap-takapuna.png", "text": "[Site2 - Takapuna] Correlation Matrix Heatmap"},
]
# page_descriptive_statistics = create_carousel_page(items_descriptive_statistics, "Correlations Heatmap with Regression")

# dashboard = vm.Dashboard(pages=[page_descriptive_statistics])

# Vizro().build(dashboard).run()


# In[27]:


from typing import List, Literal

import dash_bootstrap_components as dbc
import vizro.models as vm
from dash import html
from vizro import Vizro

## specify which elements from pydantic we are using
try:
    from pydantic.v1 import Field, PrivateAttr
except ImportError:
    from pydantic import PrivateAttr ## Fallback if specific versioning is not available

from vizro.models import Action
from vizro.models._action._actions_chain import _action_validator_factory
from vizro.models.types import capture


## 1. Create new custom component for Carousel
class Carousel(vm.VizroBaseModel):
    type: Literal["carousel"] = "carousel"
    # items: List
    items: List[dict]  # Each item in the list should be a dictionary with 'key' and 'src'
    # actions: List[Action] = []
    # List[dict]  ## Specify type of list content
    actions: List[vm.Action] = [] ## Actions that can be triggered

    _set_actions = _action_validator_factory("active_index")  

    def build(self):
        return dbc.Carousel(
            id=self.id,
            # items=self.items,
            items=[{'key': item['key'], 'src': item['src']} for item in self.items],
            controls=True,  # Show navigation controls
            indicators=True  # Show indicators
        )


## 2. Add new components to expected type - here the selector of the parent components
## Register the new component to be recognized within Vizro pages
vm.Page.add_type("components", Carousel)

## 3. Create custom action
# @capture("action")
# def carousel(active_index):
#     if active_index:
#         return "Second slide"

#     return "First slide"

@capture("action")
def handle_carousel_change(active_index: int) -> str:
    ## Proper function definition and return based on active_index
    # return f"Active slide index: {active_index}"
    return f"Slide number {active_index + 1}"

## Define a page with the custom Carousel component
# page = vm.Page(
#     title="Custom Component 2",
#     layout=vm.Layout(grid=[[i] for i in range(2)],
#                      row_min_height="500px"),
#     components=[
#         Carousel(
#             # id="carousel",
#             id="my-carousel-devtest-101",
#             items=[
#                 {"key": "1", "src": "assets/images/Descriptive-Statistics/correlations_heatmap_with_regression-1.png"},
#                 {"key": "2", "src": "assets/images/Descriptive-Statistics/correlations_heatmap_with_regression-2.png"},
#             ],
#             actions=[
#                 vm.Action(
#                     # function=carousel(),
#                     function=handle_carousel_change(),
#                     inputs=["carousel.active_index"],
#                     outputs=["carousel-card.children"]
#                 )
#             ]
#         ),
#         vm.Card(text="Click carousel to change text here.", id="carousel-card-devtest-101"),
#     ],
# )

# dashboard = vm.Dashboard(pages=[page])

# Vizro().build(dashboard).run()


# In[28]:


# import plotly.express as px
# import pandas as pd

# # Simulated data
# data = {
#     'Metric': ['RMSE', 'MSE', 'MAE', 'MAPE', 'R2', 'Adjusted R2'] * 7,
#     'Model': ['ARIMA', 'Prophet', 'NeuralProphet', 'LinearRegression', 'RandomForest', 'SVR', 'XGBoost'] * 6,
#     'Penrose_PM2.5': [4.23, 4.46, 4.98, 4.26, 4.19, 4.07, 4.66,
#                       17.91, 19.85, 24.81, 18.19, 17.60, 16.57, 21.70,
#                       3.17, 3.49, 3.76, 3.01, 3.01, 2.93, 3.41,
#                       float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'),
#                       -0.02, -0.13, -0.41, -0.03, 0.00, 0.06, -0.23,
#                       -0.02, -0.13, -0.41, -0.04, 0.00, 0.06, -0.24],
#     'Takapuna_PM2.5': [5.11, 3.77, 3.03, 4.06, 1.50, 2.40, 1.62,
#                        26.15, 14.22, 9.21, 16.50, 2.24, 5.75, 2.61,
#                        4.55, 3.34, 2.46, 1.62, 1.04, 1.98, 1.15,
#                        75.70, 85.53, 60.79, 38.08, 25.74, 52.41, 27.09,
#                        -3.75, -1.58, -0.67, -2.00, 0.59, -0.04, 0.53,
#                        -3.77, -1.59, -0.68, -2.01, 0.59, -0.05, 0.52]
# }

# df = pd.DataFrame(data)

# # Polar plot
# fig = px.bar_polar(df, r="Penrose_PM2.5", theta="Metric", color="Model",
#                    template="plotly_dark", title="Model Performance for Penrose PM2.5")

# fig.show()


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [UI/UX] Polar Bar

# In[29]:


import pandas as pd
import logging
import json

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

## Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## Load evaluation results from the provided JSON string
evaluation_results = pd.read_json('data/source/evaluation_results.json')
# logging.debug(f"Parsing evaluation results JSON data: %s", evaluation_results)
evaluation_results


# In[30]:


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


# In[31]:


## Parse the evaluation results into a DataFrame
df = parse_evaluation_results(evaluation_results)
# df

## Save the DataFrame to a CSV or Parquet file or a pickle file
df.to_csv('data/source/evaluation_results.csv', index=False)
# df.to_pickle('evaluation_results_df.pkl')
# df.to_parquet('evaluation_results_df.parquet')


# In[32]:


import os, logging
import pandas as pd              ## Data processing, file I/O
import numpy as np               ## Linear algebra

## Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
## If using pickle file or Parquet
# df = pd.read_pickle('evaluation_results_df.pkl')
# df = pd.read_parquet('evaluation_results_df.parquet')
## Load the DataFrame from the *.csv file
df = pd.read_csv('data/source/evaluation_results.csv')

## Proceed with EDA
print(df.head())


# In[33]:


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
    
    targets    = df['Target'].unique()
    models     = df['Model'].unique()
    metrics    = df['Metric'].unique()
    colorscale = px.colors.sequential.Plasma

    fig = go.Figure()

    ## Iterate over each target: Add traces for the default metric
    for target in targets:
        ## Filter DataFrame for the current target and default metric
        target_df = df[(df['Target'] == target) & (df['Metric'] == default_metric)]
        for model in models:
            model_df = target_df[target_df['Model'] == model]
            mean_value = model_df['Value'].mean()
            if len(model_df) > 0:
                text = [
                            f"Model: {model}<br>"
                            f"Metric: {default_metric}<br>"
                            f"Average {default_metric}: {mean_value:.2f}<br>" +
                            "<br>".join([f"Fold {i+1}: {model_df.iloc[i]['Value']:.2f}" for i in range(len(model_df))])
                        ],
                logging.debug(text)
                
                fig.add_trace(
                    go.Barpolar(
                        r=[mean_value],                ## Average/Mean value for the metric
                        theta=[model],                 ## Display the model names around the polar chart
                        name=f"{target}",    ## Only target in the name for legend clarity: f"{target} - {model}"
                        legendgroup=target,
                        showlegend=model == models[0], ## Only show legend for the first model to avoid repetition
                        text = text,
                        hoverinfo='text+r', ## Hover text for additional info
                    )
                )

    
    ## Set up the layout with 2/3 for the polar chart and 1/3 for the dropdown and legend
    fig.update_layout(
        title="Comparative Model Performance Across Multiple Metrics for Penrose and Takapuna PM2.5 and PM10",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, df[df['Metric'] == default_metric]['Value'].max()])
        ),
        showlegend=True,
        template='plotly_white',  ## 'plotly_white' | 'plotly_dark'
        # legend=dict(yanchor="top", y=1, xanchor="left", x=1.35),
        legend=dict(
            title="PM Targets",
            itemsizing='constant',
            # yanchor="top",
            yanchor="bottom",
            # y=1,
            xanchor="left",
            # x=1.2,
            font=dict(size=10),              ## Adjust font size for better readability
            bgcolor="rgba(255,255,255,0.7)"  ## Add a semi-transparent background for clarity
        ),
        # margin=dict(l=60, r=30, t=40, b=30),
        width=1200,  ## Adjust width to allow space for dropdown and legend
        height=800,  ## Adjust height for better layout
        # updatemenus=[
        #     {
        #         "buttons": [
        #             {
        #                 "label": metric,
        #                 "method": "update",
        #                 "args": [
        #                     {"visible": [True for _ in fig.data]},
        #                     {"title": f"Comparative Model Performance for {metric}", "showlegend": True}
        #                 ]
        #             }
        #             for metric in metrics
        #         ],
        #         "direction": "down",
        #         "showactive": False,
        #         "xanchor": "left",
        #         # "x": 0.1,
        #         # "y": 1.2,
        #     }
        # ],
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
    # for i, trace in enumerate(fig.data):  # Corrected to use enumerate for index
        value = round(trace.r[0], decimal_place)
        # if value in unique_colors:
        #     color_index = unique_colors[value]
        # else:
        #     color_index = min(unique_colors.values(), key=lambda k: abs(k - value))
        color_index = unique_colors.get(value, min(unique_colors.values(), key=lambda k: abs(k - value)))
        color_index = max(0, min(color_index, len(colorscale) - 1))
        trace.marker.color = colorscale[color_index]
        trace.marker.colorscale = colorscale  ## Apply gradient scale
        trace.marker.showscale = True         ## Ensure gradient scale is shown
        # trace.marker.showscale = True if i == 0 else False  ## Ensure gradient scale is shown only once
        # trace.marker.colorbar = dict(title=f'{default_metric} Value', tickvals=[0, 2, 4, 6, 8, 10])
        logging.debug(f"Value: {value}, Color Index: {color_index}, Color: {colorscale[color_index]}")

    ## Configured the color bar title and tick values
    fig.update_traces(marker=dict(colorbar=dict(title=f'{default_metric} Value', tickvals=[0, 2, 4, 6, 8, 10])))

    # ## Configure the color bar once
    # colorbar_config = dict(title=f'{default_metric} Value', tickvals=[0, 2, 4, 6, 8, 10])
    # ## Apply the color bar configuration only to the first trace
    # fig.data[0].marker.colorbar = colorbar_config

    logging.debug("Visualization created successfully.")
    # fig.show()
    return fig

create_visualization(df)


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [UI/UX] Performance Metrics Bar Chart

# In[34]:


## Function to calculate mean RMSE
def calculate_mean_rmse(rmse_list):
    """
    Calculate the mean of a list of RMSE values.
    
    Parameters:
    rmse_list (list): List of RMSE values.
    
    Returns:
    float: Mean RMSE value.
    """
    return sum(rmse_list) / len(rmse_list)

## Function to prepare data for visualization
def prepare_data(evaluation_results):
    """
    Prepare data for visualization from evaluation results.
    
    Parameters:
    evaluation_results (dict): Dictionary containing evaluation results.
    
    Returns:
    pd.DataFrame: DataFrame containing processed data for visualization.
    """
    data_list = []

    for target, models in evaluation_results.items():
        for model, metrics in models.items():
            avg_rmse = calculate_mean_rmse(metrics['RMSE'])
            training_time = metrics['Training Time']
            data_list.append({'Target': target, 'Model': model, 'Average RMSE': avg_rmse, 'Training Time': training_time})

    df = pd.DataFrame(data_list)
    logging.info("Data for visualization has been successfully prepared.")
    return df


# In[38]:


def create_visualization_with_highlighting(df):
    """
    Create a visualization comparing model performance and highlight the best-performing models for the same pollutant across different locations.

    * [x]  4 legend items for target variables: 
    
    Parameters:
    df (pd.DataFrame): DataFrame containing data for visualization.
    
    Returns:
    go.Figure: Plotly Figure object with the visualization.
    """
    ## Create subplots for Average RMSE and Training Time
    fig = make_subplots(rows=2, cols=1, row_heights=[0.67, 0.33], subplot_titles=('Average RMSE', 'Training Time'), shared_xaxes=False)

    ## Define a consistent color mapping for target variables
    color_map = {
        'Penrose_PM2.5':  'blue',
        'Takapuna_PM2.5': 'green',
        'Penrose_PM10':   'red',
        'Takapuna_PM10':  'purple'
    }

    ## Adding RMSE data with highlighting and setting showlegend=False to avoid duplicate legends
    for target in df['Target'].unique():
        target_data = df[df['Target'] == target]
        best_rmse = target_data['Average RMSE'].min()
        best_model = target_data[target_data['Average RMSE'] == best_rmse]['Model'].values[0]
        colors = [color_map[target] for _ in target_data['Average RMSE']]
        line_colors = ['#FFA500' if rmse == best_rmse else color_map[target] for rmse in target_data['Average RMSE']]
        print(f"Best model for {target} is {best_model} with RMSE {best_rmse:.2f}")
        fig.add_trace(go.Bar(x=target_data['Model'], y=target_data['Average RMSE'], name=target, 
                             marker_color=colors, 
                             marker_line_color=line_colors, 
                             marker_line_width=2, 
                             # marker_color=colors, 
                             showlegend=True,
                             hoverinfo='text', 
                             # text=[f"Average RMSE: <b>{rmse:.2f}</b>" for model, rmse in zip(target_data['Model'], target_data['Average RMSE'])],
                             text=[f"Average RMSE: <b>{rmse:.2f}</b>" for rmse in target_data['Average RMSE']],
                             hovertemplate=f'Target: {target}<br>' + 'Model: %{x}<br>Average RMSE: %{y:.2f}<extra></extra>'
                            ), row=1, col=1),
        fig.add_annotation(x=best_model, y=best_rmse, 
                           text=f"Best: {best_rmse:.2f}", 
                           showarrow=True, arrowhead=2, 
                           # ax=-35, ay=-25, 
                           ax=-5, ay=-5, 
                           row=1, col=1, font=dict(size=10, color=color_map[target]), bgcolor='white')

    ## Adding Training Time data with the same colors for consistency and setting showlegend=False to avoid duplicate legends
    for target in df['Target'].unique():
        target_data = df[df['Target'] == target]
        fig.add_trace(go.Bar(x=target_data['Model'], y=target_data['Training Time'], name=target, marker_color=color_map[target], showlegend=False), row=2, col=1)

    ## Update layout
    fig.update_layout(
        title_text="Model Performance Comparison: Average RMSE and Training Time",
        title_font=dict(size=20, color='black', family="Arial Black"),
        showlegend=True,
        template='plotly_white',  ## 'plotly_white' | 'plotly_dark'
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        bargap=0.3,       ## Gap between bars of adjacent locations
        bargroupgap=0.2,  ## Gap between groups of bars
        # width=1200,     ## Adjust width to allow space for dropdown and legend
        height=750,       ## Adjust height for better layout
        yaxis_title='Average RMSE (Lower is Better)',
        yaxis2_title='Training Time (Seconds)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12, color='black'),
    )

    ## Logging for debugging purposes
    logging.info("Visualization with highlighting has been successfully created.")
    return fig


# In[39]:


## Prepare data
df = prepare_data(evaluation_results)

## Create visualization with highlighting
fig = create_visualization_with_highlighting(df)

## Display the plot
fig.show()


# ![Comparative-Performance-of-All-Models](https://raw.githubusercontent.com/nnthanh101/Machine-Learning/main/docs/static/img/data-science/Comparative-Performance-of-All-Models.png)

# In[40]:


def calculate_mean_metric(metric_list):
    """
    Calculate the mean of a list of metric values.
    
    Parameters:
    metric_list (list): List of metric values.
    
    Returns:
    float: Mean metric value.
    """
    logging.debug(f"Calculating mean for metrics: {metric_list}")
    return sum(metric_list) / len(metric_list)


def prepare_data_for_markdown_tables(evaluation_results):
    """
    Prepare data for visualization from evaluation results.
    
    Parameters:
    evaluation_results (dict): Dictionary containing evaluation results.
    
    Returns:
    pd.DataFrame: DataFrame containing processed data for visualization.
    """
    data_list = []

    for target, models in evaluation_results.items():
        for model, metrics in models.items():
            avg_rmse = calculate_mean_metric(metrics['RMSE'])
            avg_mse = calculate_mean_metric(metrics['MSE'])
            avg_mae = calculate_mean_metric(metrics['MAE'])
            # avg_mape = calculate_mean_metric(metrics['MAPE'])
            avg_r2 = calculate_mean_metric(metrics['R2'])
            avg_adj_r2 = calculate_mean_metric(metrics['Adjusted R2'])
            # training_time = sum(metrics['Training Time'])
            training_time = metrics['Training Time'] if isinstance(metrics['Training Time'], list) else [metrics['Training Time']]
            training_time = sum(training_time)

            logging.info(f"Processed metrics for model {model} on target {target}: RMSE={avg_rmse}, MSE={avg_mse}, MAE={avg_mae}, R2={avg_r2}, Adjusted R2={avg_adj_r2}, Training Time={training_time}")

            data_list.append({
                'Target': target, 'Model': model, 
                'Average RMSE': avg_rmse, 'Average MSE': avg_mse, 
                'Average MAE': avg_mae, 
                # 'Average MAPE': avg_mape, 
                'Average R2': avg_r2, 'Average Adjusted R2': avg_adj_r2,
                'Training Time': training_time
            })

    df = pd.DataFrame(data_list)
    logging.info("Data for visualization has been successfully prepared.")
    return df

## Function to generate markdown tables
def generate_markdown_tables(df, targets):
    """
    Generate markdown tables for specified targets.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the processed data.
    targets (list): List of targets to generate markdown tables for.
    
    Returns:
    dict: Dictionary of markdown tables.
    """
    tables = {}
    for target in targets:
        target_data = df[df['Target'] == target]
        table_md = target_data.to_markdown(index=False)
        tables[target] = table_md
        logging.info(f"Generated markdown table for target {target}")
    return tables

## Prepare the data
df_markdown_tables = prepare_data_for_markdown_tables(evaluation_results)

## Define the targets
targets = ['Penrose_PM2.5', 'Takapuna_PM2.5', 'Penrose_PM10', 'Takapuna_PM10']

## Generate markdown tables
markdown_tables = generate_markdown_tables(df_markdown_tables, targets)

## Print the markdown tables
for target, table in markdown_tables.items():
    print(f"#### {target}")
    print(table)
    print("\n")


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [UI/UX] Feature Importances and Contributions

# In[41]:


## Extract top features for 'PM2.5': Analyzing for both sites
top_features_data11 = DataFrameAdapter.get_top_correlated_features(data=cleaned_data_site1, target='PM2.5', num_features=10)
top_features_data12 = DataFrameAdapter.get_top_correlated_features(cleaned_data_site2, 'PM2.5', num_features=10)

print("\n🌟 top_features_data11: Top 10 features highly correlated with PM2.5 in Penrose: %s\n", top_features_data11)
print("\n🌟 top_features_data12: Top 10 features highly correlated with PM2.5 in Takapuna: %s\n", top_features_data12)

## Extract top features for 'PM10': Analyzing for both sites
top_features_data21 = DataFrameAdapter.get_top_correlated_features(data=cleaned_data_site1, target='PM10', num_features=10)
top_features_data22 = DataFrameAdapter.get_top_correlated_features(cleaned_data_site2, 'PM10', num_features=10)

print("\n🌟 top_features_data21: Top 10 features highly correlated with PM10 in Penrose: %s\n", top_features_data21)
print("\n🌟 top_features_data22: Top 10 features highly correlated with PM10 in Takapuna: %s\n", top_features_data22)


# In[42]:


import joblib
import logging
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

## Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_models(model_paths):
    """
    Load models from specified paths.

    Parameters:
    model_paths (dict): Dictionary with model names as keys and paths as values.

    Returns:
    dict: Dictionary with model names as keys and loaded model objects as values.
    """
    models = {}
    try:
        for target_var, model_path in model_paths.items():
            models[target_var] = joblib.load(model_path)
            logging.info(f"Loaded model for {target_var} from {model_path}")
    except FileNotFoundError as e:
        logging.error("Model file not found: %s", e)
        raise
    except Exception as e:
        logging.error("Error loading models: %s", e)
        raise
    return models


def get_feature_importance(model, X, y):
    """Function to extract feature importance"""
    if hasattr(model, 'coef_'):
        logging.info('coef_')
        importances = np.abs(model.coef_).flatten()
    elif hasattr(model, 'feature_importances_'):
        logging.info('feature_importances_')
        importances = model.feature_importances_
    else:
        logging.info('permutation_importance')
        ## Use permutation importance as a fallback for models without built-in importance
        X_numeric = X.select_dtypes(include=[np.number])
        result = permutation_importance(model, X_numeric, y, n_repeats=10, random_state=42, n_jobs=2)
        importances = result.importances_mean

    return importances


## Model paths
model_paths = {
    "Penrose_PM2.5":  "data/models/Penrose_PM2.5_RandomForest.joblib",
    "Takapuna_PM2.5": "data/models/Takapuna_PM2.5_RandomForest.joblib",
    "Penrose_PM10":   "data/models/Penrose_PM10_RandomForest.joblib",
    # "Takapuna_PM10":  "data/models/Takapuna_PM10_SVR.joblib",
    "Takapuna_PM10":  "data/models/PM10_Takapuna_PM10_RandomForest.joblib",
}

## Load models
models = load_models(model_paths)


# In[43]:


models


# In[44]:


from sklearn.ensemble import RandomForestRegressor

def analyze_random_forest(models):
    """
    Analyze RandomForestRegressor models to extract and log feature importances and additional information.

    Parameters:
    models (dict): Dictionary with model names as keys and loaded model objects as values.
    """
    for model_name, model in models.items():
        # logging.info(f"Model {model_name} is a RandomForestRegressor")
        if isinstance(model, RandomForestRegressor):
            try:
                importances = model.feature_importances_
                n_features = model.n_features_in_
                print(f"\n[RandomForestRegressor] Model: {model_name}")
                print(f"[RandomForestRegressor] Number of features: {n_features}")
                print(f"[RandomForestRegressor] Feature importances: {importances}")
                
                ## Additional information
                logging.debug(f"[RandomForestRegressor] Number of trees: {len(model.estimators_)}")
                logging.debug(f"[RandomForestRegressor] Mean of feature importances: {importances.mean()}")
                logging.debug(f"[RandomForestRegressor] Standard deviation of feature importances: {importances.std()}")
                logging.debug(f"[RandomForestRegressor] Parameters: {model.get_params()}")
            except AttributeError as e:
                logging.error(f"Error extracting information from model {model_name}: {e}")
        else:
            logging.info(f"Model {model_name} is not a RandomForestRegressor")

## Analyze RandomForestRegressor models
analyze_random_forest(models)


# In[45]:


# ## For any tree based models (sklearn.ensemble.RandomForestRegressor)
# model = models['Penrose_PM2.5']
# importances = model.feature_importances_

# print('[RandomForestRegressor] Number of features: ', model.n_features_in_)
# print('[RandomForestRegressor] The impurity-based feature importances: ', importances)


# In[46]:


# ## sklearn.linear_model/sklearn.svm.svc
# model = models['Takapuna_PM10']

# print('[sklearn.svm.SVR] Number of features: ', model.n_features_in_)
# model.shape_fit_?

# # model.feature_names_in_
# # model.coef_.shape[-1]
# # model.n_support_


# In[47]:


import logging

def generate_markdown_table_feature_importance(top_features, feature_importances, feature_names):
    """
    Generate a markdown table summarizing feature importances and top correlated features.

    Parameters:
    top_features (list): List of tuples containing top features and their correlation with the target variable.
    feature_importances (np.array): Array of feature importances from the RandomForest model.
    feature_names (list): List of feature names corresponding to the feature importances.

    Returns:
    str: Markdown table as a string.
    """
    table = "| Feature | Correlation | Importance |\n"
    table += "|---------|-------------|------------|\n"

    # Combine top features and feature importances
    for i, (feature, corr) in enumerate(top_features):
        importance = feature_importances[feature_names.index(feature)] if feature in feature_names else "N/A"
        table += f"| {feature} | {corr:.3f} | {importance:.3f} |\n"

    return table

def analyze_and_generate_tables(models, top_features_data):
    """
    Analyze RandomForestRegressor models and generate markdown tables for feature importances.

    Parameters:
    models (dict): Dictionary with model names as keys and loaded model objects as values.
    top_features_data (dict): Dictionary with top features data for each model.
    """
    tables = {}
    for model_name, model in models.items():
        if isinstance(model, RandomForestRegressor):
            try:
                importances = model.feature_importances_
                feature_names = [f[0] for f in top_features_data[model_name]]
                table = generate_markdown_table_feature_importance(top_features_data[model_name], importances, feature_names)
                tables[model_name] = table
                print(f"\n[RandomForestRegressor] Model: {model_name}")
                print(table)
            except AttributeError as e:
                logging.error(f"Error extracting information from model {model_name}: {e}")
        else:
            logging.info(f"Model {model_name} is not a RandomForestRegressor")
    
    return tables

## Precondition: top_features_data11, top_features_data12, top_features_data21, top_features_data22 have been extracted
top_features_data = {
    'Penrose_PM2.5': top_features_data11,
    'Takapuna_PM2.5': top_features_data12,
    'Penrose_PM10': top_features_data21,
    'Takapuna_PM10': top_features_data22
}

## Analyze models and generate markdown tables
tables = analyze_and_generate_tables(models, top_features_data)

## Print the tables for copy & paste to MS Excel
for model_name, table in tables.items():
    print(f"\nMarkdown Table for {model_name}:\n{table}")


# In[ ]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

## Define a consistent color mapping for target variables
color_map = {
    'Penrose_PM2.5': 'blue',
    'Takapuna_PM2.5': 'green',
    'Penrose_PM10': 'red',
    'Takapuna_PM10': 'purple'
}

## Improved names for combo-box
model_names_map = {
    'Penrose_PM2.5': 'Penrose PM2.5',
    'Takapuna_PM2.5': 'Takapuna PM2.5',
    'Penrose_PM10': 'Penrose PM10',
    'Takapuna_PM10': 'Takapuna PM10 (RandomForest as an Alternative)'
}

## Initialize the Dash app
app = dash.Dash(__name__)

## Generate figure for feature importances
# def generate_feature_importance_figure(model_name, importances, feature_names):
def generate_feature_importance_figure(model_name, importances, feature_names, correlations):
    """
    Generate a Plotly bar chart for feature importances.

    Parameters:
    model_name (str): Name of the model.
    importances (array): Feature importances from the model.
    feature_names (list): Names of the features.
    correlations (list): Correlation values of the features.

    Returns:
    figure: Plotly figure object.
    """

    ## Combine importances, feature_names, and correlations into a single list of tuples
    features = list(zip(importances, feature_names, correlations))
    ## Sort features by importances in descending order
    features.sort(reverse=True, key=lambda x: x[0])

    ## Unzip the sorted features back into separate lists
    importances, feature_names, correlations = zip(*features)
    
    trace = go.Bar(
        x=feature_names,
        y=importances,
        # text=feature_names,
        text=[f'Correlation: {corr:.3f}' for corr in correlations],
        # marker=dict(color='rgba(55, 128, 191, 0.7)',
        marker=dict(color=color_map[model_name],
                    line=dict(color='orange', width=2))
    )
    layout = go.Layout(
        # title=f'Feature Importances for {model_name}',
        title=f'Feature Importances for {model_names_map[model_name]}',
        # title=f'Feature Importances for {model_names_map.get(model_name, model_name)}',
        xaxis=dict(title='Feature'),
        yaxis=dict(title='Importance'),
        hovermode='closest',
        template='plotly_white',  ## 'plotly_white' | 'plotly_dark'
    )
    figure = go.Figure(data=[trace], layout=layout)
    return figure

## App layout
app.layout = html.Div([
    html.H1("Feature Importances and Contributions"),
    dcc.Dropdown(
        id='model-dropdown',
        # options=[{'label': k, 'value': k} for k in models.keys()],
        options=[{'label': v, 'value': k} for k, v in model_names_map.items()],
        value='Penrose_PM2.5'
    ),
    dcc.Graph(id='feature-importance-graph')
])

## Callback to update the graph
@app.callback(
    Output('feature-importance-graph', 'figure'),
    [Input('model-dropdown', 'value')]
)
def update_graph(selected_model):
    """
    Update the feature importance graph based on the selected model.

    Parameters:
    selected_model (str): The selected model from the dropdown.

    Returns:
    figure: Plotly figure object.
    """
    model = models[selected_model]
    if isinstance(model, RandomForestRegressor):
        importances = model.feature_importances_
        feature_names = [f[0] for f in top_features_data[selected_model]]
        correlations  = [f[1] for f in top_features_data[selected_model]]
        # figure = generate_feature_importance_figure(selected_model, importances, feature_names)
        figure = generate_feature_importance_figure(selected_model, importances, feature_names, correlations)
        return figure
    else:
        logging.warning(f"Model {selected_model} is not a RandomForestRegressor.")
        return go.Figure()

## Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


# <header style="padding:3px;border-top:3px solid #E37C4D">
# 
# ## 📊 [UI/UX] Shapash

# ## Dev/Test -->

# In[ ]:





# In[ ]:


@app.callback(
    Output('feature-importance-plot', 'figure'),
    Input('target-variable-dropdown', 'value')
)
def update_feature_importance_plot(target_var):
    if target_var not in models:
        return go.Figure()

    model = models[target_var]
    try:
        if target_var == 'Penrose_PM2.5':
            X = cleaned_data_site1[top_features_data11]
        elif target_var == 'Takapuna_PM2.5':
            X = cleaned_data_site2[top_features_data12]
        elif target_var == 'Penrose_PM10':
            X = cleaned_data_site1[top_features_data21]
        else:
            X = cleaned_data_site2[top_features_data22]
        
        y = cleaned_data_site1[target_var.split('_')[1]] if 'Penrose' in target_var else cleaned_data_site2[target_var.split('_')[1]]
        # importances = get_feature_importance(model, X, y)
        importances = get_feature_importance(model, X.values, y)  # Convert X to a NumPy array to avoid warnings
        
        top_features = X.columns

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_features,
            y=importances,
            text=importances,
            textposition='auto',
            name='Feature Importance'
        ))

        fig.update_layout(
            title=f'Feature Importance for {target_var}',
            xaxis_title='Features',
            yaxis_title='Importance',
            template='plotly_white'
        )
        return fig
    except ValueError as e:
        logging.error(e)
        return go.Figure()


# In[ ]:


# Create Dash application
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Air Quality Model Feature Importance"),
    dcc.Dropdown(
        id='target-variable-dropdown',
        options=[{'label': target, 'value': target} for target in models.keys()],
        value='Penrose_PM2.5'
    ),
    dcc.Graph(id='feature-importance-plot'),
])

if __name__ == '__main__':
    app.run_server(debug=True)

# Save this script in a Python file and run it in your environment.


# ## WIP -->

# In[ ]:


# !pip install shap dash dash-bootstrap-components joblib
# !pip install shapash

import pandas as pd
import numpy as np
import logging

import os, joblib, shap
from shapash.explainer.smart_explainer import SmartExplainer
# import shapash.dashboard
from tqdm import tqdm
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from sklearn.svm import SVR

# from DataFrameAdapter import DataFrameAdapter


# In[ ]:


def extract_feature_names(feature_list):
    """
    Extracts feature names from a list of tuples.

    Args:
        feature_list (list of tuples): List where each tuple contains a feature name and its correlation value.

    Returns:
        list: A list of feature names.
    """
    try:
        return [feature[0] for feature in feature_list]
    except (TypeError, IndexError) as e:
        logging.error("Error extracting feature names: %s", e)
        return []

## Extracting feature names with error handling
top_features_data11_names = extract_feature_names(top_features_data11)
top_features_data12_names = extract_feature_names(top_features_data12)
top_features_data21_names = extract_feature_names(top_features_data21)
top_features_data22_names = extract_feature_names(top_features_data22)

## Ensure the extracted feature lists are not empty
if not top_features_data11_names or not top_features_data12_names or not top_features_data21_names or not top_features_data22_names:
    logging.error("One or more feature lists are empty. Please check the input data.")
    raise ValueError("Feature extraction failed due to empty input data.")

## Combining data for easier access
data_dict = {
    'Penrose_PM2.5': (cleaned_data_site1, 'PM2.5', top_features_data11_names, 'ARIMA(0, 1, 4)'),
    # 'Takapuna_PM2.5': (cleaned_data_site2, 'PM2.5', top_features_data12_names, 'ARIMA(0, 1, 2)'),
    # 'Penrose_PM10': (cleaned_data_site1, 'PM10', top_features_data21_names, 'ARIMA(10, 0, 0)'),
    # 'Takapuna_PM10': (cleaned_data_site2, 'PM10', top_features_data22_names, 'ARIMA(2, 0, 3)')
}
logging.debug("Data dictionary created successfully.")

## Step 1: Loading the Best Models: Load the saved models from the "data/model" folder with error handling.
try:
    models = {
        "Penrose_PM2.5":  joblib.load("data/models/Penrose_PM2.5_SVR.joblib"),
        "Takapuna_PM2.5": joblib.load("data/models/Takapuna_PM2.5_RandomForest.joblib"),
        "Penrose_PM10":   joblib.load("data/models/Penrose_PM10_SVR.joblib"),
        "Takapuna_PM10":  joblib.load("data/models/Takapuna_PM10_SVR.joblib")
    }
    logging.debug("Models loaded successfully from 'data/model' folder.")
except FileNotFoundError as e:
    logging.error("Model file not found: %s", e)
    raise
except Exception as e:
    logging.error("Error loading models: %s", e)
    raise


# In[ ]:


cleaned_data_site1[top_features_data11_names]


# In[ ]:


models


# In[ ]:


## Step 2. Calculate SHAP Values: Use SHAP to calculate feature importance for each model.
def calculate_shap_values(model, X_train, method='sample', k=100):
    """
    Calculate SHAP values for a given model and training data.

    * [x] Calculate SHAP values for SVR models using KernelExplainer
    * [x] Calculate SHAP values for RandomForest model using TreeExplainer
    * [x] Randomly select k=100 samples from the background data to use for SHAP value calculations

    Args:
        model: Trained model.
        X_train (pd.DataFrame): Training data.
        method (str): Method to reduce background samples, 'sample' or 'kmeans'.
        k (int): Number of samples to use for background data reduction.

    Returns:
        shap_values: Calculated SHAP values.
    """
    logging.debug("Calculating SHAP values for the model.")
    try:
        if method == 'sample':
            background = shap.sample(X_train, k)
        elif method == 'kmeans':
            background = shap.kmeans(X_train, k)
        else:
            raise ValueError("Invalid method for background sample reduction.")

        if isinstance(model, SVR):  ## Check for SVR
            explainer = shap.KernelExplainer(model.predict, X_train, background)
            # shap_values = explainer.shap_values(X_train)
            shap_values = explainer.shap_values(X_train, nsamples="auto", l1_reg="num_features(10)", progress_bar=tqdm)
        else:                       ## For tree-based models like RandomForest
            explainer = shap.TreeExplainer(model, background)
            # shap_values = explainer.shap_values(X_train)
            shap_values = explainer.shap_values(X_train, check_additivity=False)
        logging.info("SHAP values calculated successfully.")
        return shap_values
    except Exception as e:
        logging.error("Error calculating SHAP values: %s", e)
        raise


# In[ ]:


def save_shap_values(shap_values_dict, directory='data/shap_values'):
    """
    Save SHAP values to disk for later use.

    Args:
        shap_values_dict (dict): Dictionary containing SHAP values for each model.
        directory (str): Directory to save the SHAP values.
    """
    os.makedirs(directory, exist_ok=True)
    for key, shap_values in shap_values_dict.items():
        joblib.dump(shap_values, os.path.join(directory, f'{key}_shap_values.joblib'))
        logger.info(f"SHAP values for {key} saved successfully.")


# In[ ]:


def load_shap_values(directory='shap_values'):
    """
    Load SHAP values from disk.

    Args:
        directory (str): Directory to load the SHAP values from.

    Returns:
        dict: Dictionary containing loaded SHAP values.
    """
    shap_values_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('_shap_values.joblib'):
            key = filename.replace('_shap_values.joblib', '')
            shap_values_dict[key] = joblib.load(os.path.join(directory, filename))
            logger.info(f"SHAP values for {key} loaded successfully.")
    return shap_values_dict


# In[ ]:


## Step 2. Calculate SHAP Values Calculate SHAP values for each model and save them

## 2.1. Use SHAP to calculate feature importance for each model (with reduced background samples).
shap_values_dict = {}
for key, (data, target, features, *_) in data_dict.items():
    model = models[key]
    X_train = data[features]  ## Extracting relevant features
    shap_values_dict[key] = calculate_shap_values(model, X_train, method='sample', k=100)

## 2.2. Save SHAP values to disk
save_shap_values(shap_values_dict)


# In[ ]:


## 2.3. Load SHAP values from disk: 
shap_values_dict = load_shap_values()


# In[ ]:


## Step 3. Visualize with Shapash: using SHAP values with Shapash to create and visualize the interpretability of the models.
from shapash.explainer.smart_explainer import SmartExplainer

smart_explainers = {}
for key, shap_values in shap_values_dict.items():
    model = models[key]
    # data, target, features = data_dict[key]
    data, target, features, *_ = data_dict[key]  ## Adjust to ignore additional elements
    X_train = data[features]
    smart_explainer = SmartExplainer(model=model)
    smart_explainer.compile(x=X_train, y=data[target])
    smart_explainers[key] = smart_explainer
    logging.info(f"SmartExplainer compiled for {key}.")

## Step 4. Develop Dash Application to present the results interactively: to visualize SHAP and feature importance

## Initialize Dash app to visualize SHAP and feature importance
app = dash.Dash(__name__)

## Layout of the app
app.layout = html.Div([
    html.H1("Air Quality Model Performance and Feature Importance Analysis"),
    
    dcc.Tabs([
        dcc.Tab(label='Penrose PM2.5', children=[
            html.H2("Penrose PM2.5 & SVR Model"),
            shapash.dashboard.Dashboard(smart_explainers['Penrose_PM2.5']).run_app()
        ]),
        
        dcc.Tab(label='Takapuna PM2.5', children=[
            html.H2("Takapuna PM2.5 & Random Forest Model"),
            shapash.dashboard.Dashboard(smart_explainers['Takapuna_PM2.5']).run_app()
        ]),
        
        dcc.Tab(label='Penrose PM10', children=[
            html.H2("Penrose PM10 & SVR Model"),
            shapash.dashboard.Dashboard(smart_explainers['Penrose_PM10']).run_app()
        ]),
        
        dcc.Tab(label='Takapuna PM10', children=[
            html.H2("Takapuna PM10 & SVR Model"),
            shapash.dashboard.Dashboard(smart_explainers['Takapuna_PM10']).run_app()
        ]),
    ])
])

logging.info("Running Dash app.")
if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## WIP -->

# In[ ]:


import time
import shap
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the cleaned data
# cleaned_data_site1 = pd.read_csv('path_to_cleaned_data_site1.csv')

# Define the features and target for Penrose PM2.5 as an example
# features = ['PM10', 'PM10_Lag1', 'NOx', 'NO', 'PM10_Lag2', 'NO2', 'SO2', 'Wind_Dir', 'Air_Temp', 'Season']
target = 'PM2.5'

# Extract the relevant features and target
X = cleaned_data_site1[top_features_data11]
y = cleaned_data_site1[target]

# Load the pre-trained model
model = joblib.load("data/model/Penrose_PM2.5_SVR.joblib")

# Function to measure time for a subset
def measure_shap_time(model, X, subset_size):
    # Subsample the data
    X_subset = shap.sample(X, subset_size)
    background = X_subset

    # Measure time for SHAP calculations
    start_time = time.time()
    
    # Use KernelExplainer for SVR
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return elapsed_time

# Measure time for a small subset (e.g., 100 samples)
subset_size = 100
elapsed_time_subset = measure_shap_time(model, X, subset_size)

# Estimate time for the entire dataset (17,375 samples)
total_samples = 17375
estimated_total_time = (total_samples / subset_size) * elapsed_time_subset

print(f"Elapsed time for {subset_size} samples: {elapsed_time_subset:.2f} seconds")
print(f"Estimated total time for {total_samples} samples: {estimated_total_time:.2f} seconds")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# import os

# def calculate_and_save_shap_values(model, X_train, filename):
#     """
#     Calculate SHAP values for a given model and training data, then save to a file.

#     Args:
#         model: Trained model.
#         X_train (pd.DataFrame): Training data.
#         filename (str): The filename to save SHAP values to.
    
#     Returns:
#         shap_values: Calculated SHAP values.
#     """
#     summarized_X_train = summarize_background(X_train)
#     shap_values = calculate_shap_values(model, summarized_X_train)
#     with open(filename, 'wb') as f:
#         joblib.dump(shap_values, f)
#     logger.info(f"SHAP values saved to {filename}")
#     return shap_values



# for key, (data, target, features) in data_dict.items():
#     model = models[key]
#     X_train = data[features]
#     shap_filename = os.path.join(shap_dir, f"{key}_shap_values.joblib")
#     if os.path.exists(shap_filename):
#         shap_values = joblib.load(shap_filename)
#         logger.info(f"Loaded SHAP values from {shap_filename}")
#     else:
#         shap_values = calculate_and_save_shap_values(model, X_train, shap_filename)
#     shap_values_dict[key] = shap_values


# In[ ]:


# from shapash.explainer.smart_explainer import SmartExplainer

# smart_explainers = {}
# for key, shap_values in shap_values_dict.items():
#     model = models[key]
#     data, target, features = data_dict[key]
#     X_train = data[features]
#     smart_explainer = SmartExplainer(model=model)
#     smart_explainer.compile(x=X_train, y=data[target], shap_values=shap_values)
#     smart_explainers[key] = smart_explainer
#     logger.info(f"SmartExplainer compiled for {key}.")


# In[ ]:


# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import shapash.dashboard

# app = dash.Dash(__name__)

# app.layout = html.Div([
#     html.H1("Model Performance and Feature Importance Analysis"),
    
#     dcc.Tabs([
#         dcc.Tab(label='Penrose PM2.5', children=[
#             html.H2("Penrose PM2.5 - SVR Model"),
#             shapash.dashboard.Dashboard(smart_explainers['Penrose_PM2.5']).run_app()
#         ]),
        
#         dcc.Tab(label='Takapuna PM2.5', children=[
#             html.H2("Takapuna PM2.5 - Random Forest Model"),
#             shapash.dashboard.Dashboard(smart_explainers['Takapuna_PM2.5']).run_app()
#         ]),
        
#         dcc.Tab(label='Penrose PM10', children=[
#             html.H2("Penrose PM10 - SVR Model"),
#             shapash.dashboard.Dashboard(smart_explainers['Penrose_PM10']).run_app()
#         ]),
        
#         dcc.Tab(label='Takapuna PM10', children=[
#             html.H2("Takapuna PM10 - SVR Model"),
#             shapash.dashboard.Dashboard(smart_explainers['Takapuna_PM10']).run_app()
#         ]),
#     ])
# ])

# logger.info("Running Dash app.")
# if __name__ == '__main__':
#     app.run_server(debug=True)

