import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Creating the output paths when not existing
to_create_paths = ['plots/SalesForecast']
for my_path in to_create_paths:
    if not os.path.exists(os.path.join(os.getcwd(),my_path)):
        # Create the directory
        os.makedirs(os.path.join(os.getcwd(),my_path))

########################################################################################################################
# -First part plots the sales forecast comparison figure with qty Sold
# -Second part makes the salesforecast comparison figure with the qty Sold but now using the csv file for all Europe data
# Normally this plot should have been made per product hierarchy but the hive table was not yet ready (only the full
# Europe market could be done)
#
data = [
    ['202206',1234993,1753817],
    ['202207',1144961,1743527],
    ['202208',1187915,1623436],
    ['202209',1376859,1791373],
    ['202210',1433190,2035717],
    ['202211',1508620,2063908],
    ['202212',1052453,1484703],
    ['202301',1139093,1503235],
    ['202302',1129050,1437713],
    ['202303',1224868,1586490],
    ['202304',995450,1354428],
    ['202305',1069391,1411502],
    ['202306',1116571,1433185]
]
df = pd.DataFrame(data, columns=['date', 'Actual', 'Forecast'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,10), sharex=True)
ax1.errorbar(df['date'], df['Actual'],  marker='o', color='k',linestyle="None",label='Actual Sales')
ax1.errorbar(df['date'], df['Forecast'],  marker='o', color='r',linestyle="None",label='Forecast Sales')
ax1.set_xticklabels([])
ax1.set_ylabel('Sales')
ax1.set_title('Sales + Forcast (June 2022 - June 2023)')
ax1.legend()
ax1.grid()

vol_diff = (df['Actual']-df['Forecast']) / df['Actual'] *100
pos_vol_diff =  [val if val > 0 else 0 for val in vol_diff]
neg_vol_diff =  [val if val < 0 else 0 for val in vol_diff]

ax2.bar(df['date'],pos_vol_diff,  color='green')
ax2.bar(df['date'],neg_vol_diff,  color='red')
ax2.set_ylim([-55,10])
ax2.set_xticks(df['date'])
ax2.set_xticklabels(df['date'], rotation=90)
ax2.set_ylabel('Percentage Error (A-F)/A*100')
ax2.grid()
plt.savefig(os.path.join('plots','SalesForecast','Sales_Forcast_22-23.png'))

########################################################################################################################
# Second part starts here
# Prasanna generated the csv-file with actual qtySold and Forecasted qtySold
# (874 PNCs are missing in qes.product_properties_base table but present in BC Sales Volume Forecast data.)
#
#  A) No scale factor applied
#
df = pd.read_csv(os.path.join("data","actual_vs_forecast_sales_volume_all_europe.csv"), sep=';')
df = df.rename(columns={'period': 'date', 'actual_qty_sold': 'Actual', 'pred_qty_sold': 'Forecast'})

df['date'] = df['date'].astype(str)
df = df.iloc[1:]
#scale_factor = df['Actual'].mean()/df['Forecast'].mean()
scale_factor = 1
mape = np.mean(np.abs((df['Actual']-df['Forecast']*scale_factor) / df['Actual'] *100))
mape_std = np.std(np.abs((df['Actual']-df['Forecast']*scale_factor) / df['Actual'] *100))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,6), sharex=True)
#ax1.errorbar(df['date'], df['Actual'],  marker='o', color='k',linestyle="None",label='Actual Sales')
ax1.bar(df['date'], df['Actual'], color='blue', alpha=0.7, label='Actual Sales')
ax1.errorbar(df['date'], df['Forecast']*scale_factor,  marker='o', color='r',label='Forecast Sales')
ax1.set_xticklabels([])
ax1.set_ylabel('Sales')
title_txt = 'All Product Lines for Europe (Sep 2019 - Sep 2023)\nWith scale factor={:.3f} MAPE={:.2f}+-{:.3f}'.format(scale_factor, mape, mape_std)
ax1.set_title(title_txt)
ax1.legend()
ax1.grid()

vol_diff = (df['Actual']-df['Forecast']*scale_factor) / df['Actual'] *100
pos_vol_diff =  [val if val > 0 else 0 for val in vol_diff]
neg_vol_diff =  [val if val < 0 else 0 for val in vol_diff]

ax2.bar(df['date'],pos_vol_diff,  color='green')
ax2.bar(df['date'],neg_vol_diff,  color='red')
ax2.set_ylim([-35,35])
ax2.set_xticks(df['date'])
ax2.set_xticklabels(df['date'], rotation=90)
ax2.set_ylabel('Percentage Error [%]')
ax2.grid()
plt.tight_layout()
plt.savefig(os.path.join('plots','SalesForecast','Sales_Forcast_sfact{:.1f}.png'.format(scale_factor)))
#
#  B) With optimal scale factor
#
df['date'] = df['date'].astype(str)
df = df.iloc[1:]
scale_factor = df['Actual'].mean()/df['Forecast'].mean()
mape = np.mean(np.abs((df['Actual']-df['Forecast']*scale_factor) / df['Actual'] *100))
mape_std = np.std(np.abs((df['Actual']-df['Forecast']*scale_factor) / df['Actual'] *100))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,6), sharex=True)
#ax1.errorbar(df['date'], df['Actual'],  marker='o', color='k',linestyle="None",label='Actual Sales')
ax1.bar(df['date'], df['Actual'], color='blue', alpha=0.7, label='Actual Sales')
ax1.errorbar(df['date'], df['Forecast']*scale_factor,  marker='o', color='r',label='Forecast Sales')
ax1.set_xticklabels([])
ax1.set_ylabel('Sales')
title_txt = 'All Product Lines for Europe (Sep 2019 - Sep 2023)\nWith scale factor={:.3f} MAPE={:.2f}+-{:.3f}'.format(scale_factor, mape, mape_std)
ax1.set_title(title_txt)
ax1.legend()
ax1.grid()

vol_diff = (df['Actual']-df['Forecast']*scale_factor) / df['Actual'] *100
pos_vol_diff =  [val if val > 0 else 0 for val in vol_diff]
neg_vol_diff =  [val if val < 0 else 0 for val in vol_diff]

ax2.bar(df['date'],pos_vol_diff,  color='green')
ax2.bar(df['date'],neg_vol_diff,  color='red')
ax2.set_ylim([-35,35])
ax2.set_xticks(df['date'])
ax2.set_xticklabels(df['date'], rotation=90)
ax2.set_ylabel('Percentage Error [%]')
ax2.grid()
plt.tight_layout()
plt.savefig(os.path.join('plots','SalesForecast','Sales_Forcast_sfact{:.1f}.png'.format(scale_factor)))