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
        return (data, None)

    newCasesLog = pd.DataFrame(np.log10(data.NewConfirmed))
    cumCasesLog = pd.DataFrame(np.log10(data.CumConfirmed))
    newCasesLog.columns = [column.replace('Confirmed', 'ConfirmedLog') for column in newCasesLog.columns]
    cumCasesLog.columns = [column.replace('Confirmed', 'ConfirmedLog') for column in cumCasesLog.columns]
    data = data.join(newCasesLog)
    data = data.join(cumCasesLog)

    (data, params) = FitLine(data)
    return (data, params)

def FitLine(countryData):
    X = np.asarray(countryData["CumConfirmedLog"])
    y = np.asarray(countryData["NewConfirmedLog"])
    lastElem = np.argmax(y)
    Xt = X[:lastElem+1]
    yt = y[:lastElem+1]
    if len(yt) < 2:
        countryData['Predictions'] = countryData["NewConfirmed"]
        return (countryData, None)

    model = sm.RLM(yt, Xt, M=sm.robust.norms.HuberT()).fit()
    predictions = model.predict(X)

    whereLarger = np.argwhere(y > predictions)
    if whereLarger.size == 0:
        countryData['Predictions'] = countryData["NewConfirmed"]
        return (countryData, None)

    lastElem = whereLarger[-1,0]
    Xt = X[:lastElem+1]
    yt = y[:lastElem+1]
    if len(yt) < 2:
        countryData['Predictions'] = countryData["NewConfirmed"]
        return (countryData, None)

    model = sm.RLM(yt, Xt, M=sm.robust.norms.HuberT()).fit()
    predictions = np.concatenate((pow(10, model.predict(Xt)), np.asarray(countryData["NewConfirmed"])[lastElem+1:]), axis=0)

    predictionsColumn = pd.DataFrame(predictions, columns = ['Predictions'])
    countryData = countryData.reset_index(drop=True)
    countryData = pd.concat([countryData, predictionsColumn], axis=1)
    return (countryData, model.params)

def AddCountryToPlot(ax, color, countryData, country, marker = '.'):
    if countryData['CumConfirmed'].iloc[-1] > 1000:
        # ax.plot(countryData.CumConfirmed.array.to_numpy(), countryData.NewConfirmed.array.to_numpy(), marker, c=color)
        ax.plot(countryData.CumConfirmed.array.to_numpy(), countryData['Predictions'].array.to_numpy(), '-', c=color, label=country)

slopes = []
for country in countries:
    (data, params) = GetCountryData(country)
    if (params is not None) and (data.CumConfirmed.iloc[-1] > 1000):
        slopes.append(params[0])
slopeMean = np.mean(slopes)
slopeStd = np.std(slopes)

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
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

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
            style={'height':'800px'},
            config={ 'displayModeBar': False }
        ),
    ]
)

def chart(data, cname):
    x = np.linspace(0.1, 5.7, 2)
    x_rev = x[::-1]
    y = pow(10, x * slopeMean)

    fig = go.Figure()

    for fold in range(1, 4, 1):
        y_upper = pow(10, x * (slopeMean + fold * slopeStd))
        y_lower = pow(10, x_rev * (slopeMean - fold * slopeStd))

        fig.add_trace(go.Scatter(
            x=pow(10, np.concatenate((x, x_rev))),
            y=np.concatenate((y_upper, y_lower)),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line_color='rgba(255,255,255,0)',
            # showlegend=False,
            mode='lines',
            name='Mean country rate - '+str(fold)+'STD',
        ))

    fig.add_trace(go.Scatter(
        x=pow(10, x), y=y,
        line_color='rgb(0,100,80)',
        line_width=2,
        mode='lines',
        name='Mean country rate',
    ))
    fig.add_trace(go.Scatter(
        x=data.CumConfirmed, y=data.NewConfirmed,
        mode='markers',
        marker_color='rgb(231,107,243)',
        name=cname+' - data'
    ))
    fig.add_trace(go.Scatter(
        x=data.CumConfirmed, y=data.Predictions,
        mode='lines+markers',
        marker_color='rgb(231,107,243)',
        name=cname+' - model'
    ))

    fig.update_xaxes(type="log", title="Cumulative number of infected people.") \
       .update_yaxes(type="log", title="Number of daily new infections.")

    return fig

@app.callback(
    Output('plot_country', 'figure'), 
    [Input('country', 'value')]
)
def update_plot_country(country):
    (data, params) = GetCountryData(country)
    return chart(data, country)

if __name__ == '__main__':
    app.run_server(host="127.0.0.1")