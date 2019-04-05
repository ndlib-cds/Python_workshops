# clear all
%reset -f

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
import os

# get current directory
os.getcwd()

# set working directory
os.chdir('/Users/jng2/Dropbox/Work/Library/CDS/Python pandas')

dfauto = pd.read_csv('auto.csv')
dfauto
dfauto.head()
dfauto.dtypes
dfauto.columns.values
dfauto.describe()

# split make into two parts, store as new columns

dfauto['make'] = dfauto['vehicle'].str.split().str[0]

dfauto['model'] = dfauto['vehicle'].str.split().str[1:]
dfauto['model'] = dfauto['model'].apply(' '.join)

# move vehicle, make and model to front
cols = list(dfauto.columns)
cols = [cols[-1]] + cols[:-1]
cols = [cols[-1]] + cols[:-1]
dfauto = dfauto[cols]

# compute average price and mpg by make
aveprice = dfauto['price'].groupby(dfauto['make']).mean()
aveprice.name = 'ave_price'

avempg = dfauto['mpg'].groupby(dfauto['make']).mean()
avempg.name = 'ave_mpg'

# store as new dataframe
dfmeans = pd.concat([aveprice, avempg], axis='columns')

dfmeans['make'] = dfmeans.index

# remove row names and commit change in place
dfmeans.reset_index(drop=True, inplace=True)

# scatter plot of ave_price vs ave_mpg
dfmeans.plot(x='ave_price', y='ave_mpg', kind='scatter')

# scatter plot with labeled points
fig, ax=plt.subplots()
ax.scatter( dfmeans['ave_price'], dfmeans['ave_mpg'])
for i, row in dfmeans.iterrows():
    ax.annotate( row['make'], xy=(row['ave_price'], row['ave_mpg']))

# save figure to working directory
plt.savefig('scatterplot.png')

# back to original data for more exploration
dfauto['foreign'].value_counts()
dfauto['foreign'].value_counts().plot(kind='bar')

# create dummy variable for good fuel economy
dfauto['goodmpg'] = dfauto['mpg'] >= 30

# cross tabulate
pd.crosstab(dfauto['foreign'], dfauto['goodmpg'])

# identify vehicles with good mpg
dfauto.loc[ dfauto['goodmpg']==True ]
dfauto.loc[ dfauto['goodmpg']==True, 'make' ]

# regression
model = sm.ols(formula="price ~ mpg + foreign", data=dfauto).fit()
print(model.summary())

# reshape data from wide to long
df_bpwide = pd.read_csv('bpwide.csv')

df_bplong = pd.wide_to_long(df_bpwide, 'bp', i='patient', j='when', 
                         sep='_', 
                         suffix='.')

df_bplong.reset_index(inplace=True)

# sort data by patient
df_bplong = df_bplong.sort_values('patient')

# merging two datasets
dfhappy = pd.read_csv('happy_annual.csv')
dfgdp = pd.read_csv('gdp_annual.csv')
df_happy_gdp = pd.merge(dfhappy, dfgdp)

# plot gdp and happiness over time (two y axes)
fig, ax1 = plt.subplots()
line1 = ax1.plot( df_happy_gdp['year'], df_happy_gdp['happy'])
ax2 = ax1.twinx()
line2 = ax2.plot( df_happy_gdp['year'], df_happy_gdp['gdppc'], 
                 linestyle='dashed',
                 color='green')
# add legend
line1, label1 = ax1.get_legend_handles_labels()
line2, label2 = ax2.get_legend_handles_labels()
ax2.legend(line1 + line2, label1 + label2)
plt.savefig('gdphappytrends.png')
