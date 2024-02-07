#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess

get_ipython().system('pip install pandas dash')


# In[2]:


get_ipython().run_line_magic('pip', 'install pandas dash')


# In[3]:


get_ipython().system('wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"')


# In[4]:


import requests

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"
response = requests.get(url)


# In[5]:


csv_content = response.content
import pandas as pd
df = pd.read_csv(pd.io.common.StringIO(csv_content.decode('utf-8')))
df.head()
distinct_values = df['Launch Site'].unique()
print(distinct_values)
df.head()


# In[6]:


#url1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_3/spacex_dash_app.py"
#response1 = requests.get(url1)


# In[ ]:





# In[7]:


#with open("spacex_dash_app.py", "w") as file:
#    file.write(response1.text)


# In[8]:


#response1.text


# In[9]:


get_ipython().system('pip install wget')


# In[10]:


import wget


# In[11]:


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"
response = requests.get(url)

with open("spacex_launch_dash.csv", "wb") as file:
    file.write(response.content)


# In[12]:


#%run spacex_dash_app.py


# In[ ]:



    


# In[ ]:


# %load spacex_dash_app.py
# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                # dcc.Dropdown(id='site-dropdown',...)
                                html.Br(),
                                dcc.Dropdown(id='site-dropdown',
                options=[
                    {'label': 'All Sites', 'value': 'ALL'},
                    {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                    {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
                    {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                    {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'},
                    {'label': 'PandhiKokku Balayya', 'value': 'site4'},
                ],
                value='ALL',
                placeholder="place holder here",
                searchable=True
                ),


                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                              

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                  html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                dcc.RangeSlider(id='payload-slider',
                                min=0, max=10000, step=1000,
                                marks={0: '0',
                                100: '100'},
                                value=[min_payload,max_payload])
                                ])
@app.callback(
    [Output(component_id='success-pie-chart', component_property='figure'),
     Output(component_id='success-payload-scatter-chart', component_property='figure')],
    [Input(component_id='site-dropdown', component_property='value'),
     Input(component_id='payload-slider', component_property='value')]
)
def update_charts(entered_site, payload_range):
    filtered_df = spacex_df

    # Filter by selected site
    if entered_site != 'ALL':
        filtered_df = filtered_df[filtered_df['Launch Site'] == entered_site]

    # Filter by payload range
    filtered_df = filtered_df[(filtered_df['Payload Mass (kg)'] >= payload_range[0]) &
                              (filtered_df['Payload Mass (kg)'] <= payload_range[1])]

    # Pie chart for success vs. failure
    pie_chart = px.pie(filtered_df, names='class', title=f'Success vs. Failure for {entered_site}')

    # Scatter chart for payload vs. launch success
    scatter_chart = px.scatter(
        filtered_df,
        x='Payload Mass (kg)',
        y='class',
        color='Booster Version Category',
        title='Payload vs. Launch Success',
        labels={'class': 'Launch Outcome'}
    )

    return pie_chart, scatter_chart


        
       
        
        


                                  
                                
                                
    
        

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output


# Run the app
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8005)
   
    


# In[ ]:


with open("spacex_dash_app.py", "r") as file:
    code_content = file.read()

code_content


# In[ ]:


spacex_df.head()


# In[ ]:


filtered_df.head()


# In[ ]:




