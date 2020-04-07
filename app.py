import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import statsmodels.api as sm
from sklearn import datasets
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output

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

def GetCountryData(country, state="<all>"):
    data = allData.loc[allData['Country/Region'] == country] \
                  .drop('Country/Region', axis=1)
    if state == '<all>':
        data = data.drop('Province/State', axis=1) \
                   .groupby("date") \
                   .sum() \
                   .reset_index()
    else:
       data = data.loc[data['Province/State'] == state]
    data = data.drop(['date'], axis=1)
    newCases = data.select_dtypes(include='Int64').diff().fillna(0)
    newCases.columns = [column.replace('Cum', 'New') 
                        for column in newCases.columns]
    data = data.join(newCases)
    data = data[data["CumConfirmed"] != 0]
    data = data[data["NewConfirmed"] != 0]

    if data["CumConfirmed"].iloc[-1] < 100:
        data['Predictions'] = data['NewConfirmed']
        return data

    newCasesLog = pd.DataFrame(np.log10(data.NewConfirmed))
    cumCasesLog = pd.DataFrame(np.log10(data.CumConfirmed))
    newCasesLog.columns = [column.replace('Confirmed', 'ConfirmedLog') for column in newCasesLog.columns]
    cumCasesLog.columns = [column.replace('Confirmed', 'ConfirmedLog') for column in cumCasesLog.columns]
    data = data.join(newCasesLog)
    data = data.join(cumCasesLog)

    data = FitLine(data)
    return data

def FitLine(countryData):
    X = np.asarray(countryData["CumConfirmedLog"])
    y = np.asarray(countryData["NewConfirmedLog"])
    lastElem = np.argmax(y)
    Xt = X[:lastElem+1]
    yt = y[:lastElem+1]
    if len(yt) < 2:
        countryData['Predictions'] = countryData["NewConfirmed"]
        return countryData

    model = sm.RLM(yt, Xt, M=sm.robust.norms.HuberT()).fit()
    predictions = model.predict(X)

    whereLarger = np.argwhere(y > predictions)
    if whereLarger.size == 0:
        countryData['Predictions'] = countryData["NewConfirmed"]
        return countryData

    lastElem = whereLarger[-1,0]
    Xt = X[:lastElem+1]
    yt = y[:lastElem+1]
    if len(yt) < 2:
        countryData['Predictions'] = countryData["NewConfirmed"]
        return countryData

    model = sm.RLM(yt, Xt, M=sm.robust.norms.HuberT()).fit()
    predictions = np.concatenate((pow(10, model.predict(Xt)), np.asarray(countryData["NewConfirmed"])[lastElem+1:]), axis=0)

    predictionsColumn = pd.DataFrame(predictions, columns = ['Predictions'])
    countryData = countryData.reset_index(drop=True)
    countryData = pd.concat([countryData, predictionsColumn], axis=1)
    return countryData

def AddCountryToPlot(ax, color, countryData, country, marker = '.'):
    if countryData['CumConfirmed'].iloc[-1] > 1000:
        # ax.plot(countryData.CumConfirmed.array.to_numpy(), countryData.NewConfirmed.array.to_numpy(), marker, c=color)
        ax.plot(countryData.CumConfirmed.array.to_numpy(), countryData['Predictions'].array.to_numpy(), '-', c=color, label=country)


## Plot with matplotlib.
# color=iter(cm.rainbow(np.linspace(0,1,len(countries))))
# fig = plt.figure()
# ax = plt.gca()
# for i in range(len(countries)):
#     AddCountryToPlot(ax, next(color), GetCountryData(countries[i]), countries[i], '.')
#     ax.set_yscale('log')
#     ax.set_xscale('log')
# ax.legend()
# plt.show()

## Interested countries
# Serbia, China, Italy, US, South Korea, Russia, Sweden, Austria, Australia

## Plot with dash and plotly
external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
tickFont = {'size':12, 'color':"rgb(30,30,30)", \
            'family':"Courier New, monospace"}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    style={ 'font-family':"Courier New, monospace" },
    children=[
        html.H1('Is your country getting better w.r.t. Coronavirus (COVID-19)?'),
        html.Div(className="row", children=[
            html.Div(className="selector", children=[
                html.H5('Country'),
                dcc.Dropdown( 
                    id='country',
                    options=[{'label':c, 'value':c} \
                        for c in countries],
                    value='Serbia'
                )
            ]),
        ]),
        dcc.Graph(
            id="plot_country",
            config={ 'displayModeBar': False }
        ),
    ]
)

def chart(data):
    figure = go.Figure(data=[
        go.Scatter(
            x=data.CumConfirmed, y=data.Predictions,
            mode='lines+markers',
            marker_color='rgb(255,0,0)',
        ),
        go.Scatter(
            x=data.CumConfirmed, y=data.NewConfirmed,
            mode='markers',
            marker_color='rgb(255,0,0)'
        ),
    ])
    figure.update_layout(
                legend=dict(x=.05, y=0.95), 
                plot_bgcolor='#FFFFFF', font=tickFont) \
          .update_xaxes(
              title="x axis title",
              type="log", dtick=1,
              showgrid=True, gridcolor='#DDDDDD') \
          .update_yaxes(
              title="yaxisTitle", showgrid=True, gridcolor='#DDDDDD', dtick=1, type="log")
    return figure

@app.callback(
    Output('plot_country', 'figure'), 
    [Input('country', 'value')]
)
def update_plot_country(country):
    data = GetCountryData(country)
    return chart(data)

if __name__ == '__main__':
    app.run_server(host="127.0.0.1")