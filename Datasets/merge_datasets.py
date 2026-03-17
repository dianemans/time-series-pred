##################################################################################################
#
# This script merges the historical returns dataset (2000-2018) with the recent returns dataset 
# (2019-2025) to create a complete dataset of returns from 2000 to 2025.
#
##################################################################################################


import pandas as pd

df_historique = pd.read_csv("Datasets/230216_returns.csv")
df_historique['Date'] = pd.to_datetime(df_historique['Date'])
df_historique = df_historique.set_index('Date')

df_recent = pd.read_csv("Datasets/230216_returns_19_25.csv")
df_recent['Date'] = pd.to_datetime(df_recent['Date'])
df_recent = df_recent.set_index('Date')

# Concatenation
df_complet = pd.concat([df_historique, df_recent])

df_complet = df_complet.sort_index() # to sure they are in chronological order
df_complet = df_complet[~df_complet.index.duplicated(keep='last')] # avoid duplicates

df_complet.to_csv("Datasets/returns_all.csv")