import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd
import matplotlib.pyplot as plt

baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
def loadData(fileName, columnName):
    data = pd.read_csv(baseURL + fileName) \
             .drop(['Lat', 'Long'], axis=1) \
             .melt(id_vars=['Province/State', 'Country/Region'], 
                 var_name='date', value_name=columnName) \
             .astype({'date':'datetime64[ns]', columnName:'Int64'}, 
                 errors='ignore')
    data['Province/State'].fillna('<all>', inplace=True)
    data[columnName].fillna(0, inplace=True)
    return data

allData = loadData("time_series_covid19_confirmed_global.csv", "CumConfirmed")

countries = allData['Country/Region'].unique()
countries.sort()

def nonreactive_data(country, state):
    data = allData.loc[allData['Country/Region'] == country] \
                  .drop('Country/Region', axis=1)
    if state == '<all>':
        data = data.drop('Province/State', axis=1) \
                   .groupby("date") \
                   .sum() \
                   .reset_index()
    else:
       data = data.loc[data['Province/State'] == state]
    newCases = data.select_dtypes(include='Int64').diff().fillna(0)
    newCases.columns = [column.replace('Cum', 'New') 
                        for column in newCases.columns]
    data = data.join(newCases)
    data['dateStr'] = data['date'].dt.strftime('%b %d, %Y')
    return data

serbia = nonreactive_data('Serbia', '<all>')
china = nonreactive_data('China', '<all>')
italy = nonreactive_data('Italy', '<all>')
us = nonreactive_data('US', '<all>')
russia = nonreactive_data('Russia', '<all>')
skorea = nonreactive_data('Korea, South', '<all>')
sweden = nonreactive_data('Sweden', '<all>')

fig = plt.figure()
ax = plt.gca()

ax.plot(serbia.CumConfirmed.array.to_numpy(), serbia.NewConfirmed.array.to_numpy(), '.-')
ax.plot(china.CumConfirmed.array.to_numpy(), china.NewConfirmed.array.to_numpy(), '.')
ax.plot(italy.CumConfirmed.array.to_numpy(), italy.NewConfirmed.array.to_numpy(), '.')
ax.plot(us.CumConfirmed.array.to_numpy(), us.NewConfirmed.array.to_numpy(), '.')
ax.plot(sweden.CumConfirmed.array.to_numpy(), sweden.NewConfirmed.array.to_numpy(), '.-')
# ax.plot(russia.CumConfirmed.array.to_numpy(), russia.NewConfirmed.array.to_numpy(), '.-')
# ax.plot(skorea.CumConfirmed.array.to_numpy(), skorea.NewConfirmed.array.to_numpy(), '.-')

ax.set_yscale('log')
ax.set_xscale('log')
plt.show()