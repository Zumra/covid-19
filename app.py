import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import statsmodels.api as sm
from sklearn import datasets

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

color=iter(cm.rainbow(np.linspace(0,1,len(countries))))

fig = plt.figure()
ax = plt.gca()
for i in range(len(countries)):
    AddCountryToPlot(ax, next(color), GetCountryData(countries[i]), countries[i], '.')
    ax.set_yscale('log')
    ax.set_xscale('log')

ax.legend()
plt.show()

# AddCountryToPlot(ax, next(color), GetCountryData('Serbia'), 'o')
# AddCountryToPlot(ax, next(color), GetCountryData('China'))
# AddCountryToPlot(ax, next(color), GetCountryData('Italy'))
# AddCountryToPlot(ax, next(color), GetCountryData('US'))
# AddCountryToPlot(ax, next(color), GetCountryData('Korea, South'))
# AddCountryToPlot(ax, next(color), GetCountryData('Russia'))
# AddCountryToPlot(ax, next(color), GetCountryData('Sweden'))
# AddCountryToPlot(ax, next(color), GetCountryData('Croatia'), 'x')