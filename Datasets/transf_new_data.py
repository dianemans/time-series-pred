##################################################################################################
#
# This script process the raw data from 2019 to 2025 to format it as the returns we already have
#
##################################################################################################

import pandas as pd
import numpy as np

df = pd.read_excel("Datasets/tout px last 19-25.xlsx")
df = df.rename(columns={'Unnamed: 0': 'Date'})

data_dictionnary = pd.read_excel("Datasets/230210 Factor Set - Dictionnary (bbg only).xlsx")
data_dictionnary.index = data_dictionnary["SYMBOL"]
data_dictionnary = data_dictionnary.drop("SYMBOL", axis=1)

# CLean the head by suppressing the first line and setting the date as index
df = df.iloc[1:].copy()
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Convert to numerical values
df = df.replace('#N/A N/A', np.nan)
df = df.replace('#N/A Invalid Security', np.nan)
df = df.astype(float)

# Forward fill
df = df.ffill(limit=21*3)

# returns computation
returns = df.replace(0, np.nan).sub(df.shift(1)).div(df.shift(1).abs()) # avoid look-ahead bias
returns = returns.replace(0, np.nan)

#Particular case of rates
rates_columns = data_dictionnary['Nature'].where(data_dictionnary['Nature'] == 'Rate').dropna().index
rates_columns = [c for c in rates_columns if c in df.columns]
returns[rates_columns] = (df / 100).diff()[rates_columns]

# Particular case of volatility indices
volatility_columns = ["UX1 Index", "VXEEM Index", "VXEFA Index", "VXEWZ Index",
                      "VXFXI Index", "VXGDX Index", "VXSLV Index"]
vol_cols_exist = [c for c in volatility_columns if c in df.columns]
returns[vol_cols_exist] = df[vol_cols_exist]

# final cleaning
returns = returns.replace(0, np.nan).dropna(how='all')
returns = returns.replace(0, np.nan).dropna(how='all', axis=1)
returns = returns.replace(np.inf, np.nan)
returns = returns.replace(-np.inf, np.nan)

# aberrant returns taken from the origin polymodels_data notebook
aberrant_returns = [
    ("2004-03-23", "F3DIND Index"), ("2018-06-04", "MXEG0IN Index"), ("2001-06-20", "SASEELE Index"),
    ("2000-01-12", "SASETISI Index"), ("2010-08-16", "JMDEUR Curncy"), ("2001-12-20", "JFINX Index"),
    ("2009-12-02", "MXCO0CS Index"), ("2011-11-17", "JMDEUR Curncy"), ("2018-03-01", "MXEG0CS Index"),
    ("2014-03-03", "MXDK0MT Index"), ("2009-11-30", "F3REAL Index"), ("2016-01-04", "SASETISI Index"),
    ("2005-05-02", "MXID0EN Index"), ("2003-04-01", "MXZA0EN Index"), ("2007-02-27", "DWGFN Index"),
    ("2010-12-02", "MXBE0IN Index"), ("2002-07-25", "SASEELE Index"), ("2002-07-24", "SASETCSI Index"),
    ("2012-11-23", "NGSEFB10 Index"), ("2011-08-02", "MXDK0CD Index"), ("2022-03-10", "JMSMX Index"),
    ("2002-05-28", "RGUSMS Index"), ("2022-02-28", "MXRU0CS Index"), ("2019-08-12", "MXAR0FN Index"),
    ("2019-08-12", "MXAR0UT Index"), ("2004-09-27", "XUMAL Index"), ("2010-11-30", "MXBR0HC Index"),
    ("2015-12-01", "MXAR0CS Index"), ("2002-10-02", "F3INFT Index"), ("2004-08-05", "BOXBBC9V Index"),
    ("2004-09-07", "BOXBBC9V Index"), ("2004-12-20", "DWMFOG Index"), ("2022-02-24", "MXRU0CD Index"),
    ("2016-10-19", "F3INFT Index"), ("2005-06-03", "BOXBBC9V Index"), ("2001-11-15", "MXNZ0HC Index"),
    ("2008-04-24", "JMDEUR Curncy"), ("2019-07-24", "EGXREAL Index"), ("2019-04-30", "EGXREAL Index"),
    ("2019-05-02", "EGXREAL Index"), ("2019-07-02", "EGXREAL Index"), ("2005-06-01", "MXEG0MT Index"),
    ("2008-11-17", "MXRU0CD Index"), ("2008-03-25", "MXEG0IN Index"), ("2019-02-04", "EGXREAL Index"),
    ("2014-02-17", "NGSEOG5 Index"), ("2005-05-27", "BOXBBC9V Index"), ("2011-03-01", "MXDK0CD Index"),
    ("2019-08-14", "EGXREAL Index"), ("2019-06-10", "EGXREAL Index"), ("2011-10-11", "NGSEOG5 Index"),
    ("2011-10-12", "NGSEIN10 Index"), ("2008-09-02", "BOXBBC9V Index"), ("2005-05-03", "BOXBBC9V Index"),
    ("2008-09-09", "BOXBBC9V Index"), ("2002-01-17", "MXAR0FN Index"), ("2003-09-18", "BOXBBC9V Index"),
    ("2004-01-05", "BOXBBC9V Index"), ("2002-05-28", "RGUSSS Index"), ("2002-05-27", "RGUSSS Index"),
    ("2008-12-12", "NGSEFB10 Index"), ("2011-01-31", "BOXBBC9V Index"), ("2011-08-05", "BOXBBC9V Index"),
    ("2002-07-02", "SASETISI Index"), ("2016-11-15", "SENCYX Index"), ("2018-06-01", "MXAR0CD Index"),
    ("2000-08-03", "FNMR Index"), ("2000-12-25", "JN1 Comdty"), ("2016-09-19", "BOXBBC9V Index"),
    ("2000-05-23", "BSETCD Index"), ("2016-06-21", "NGNJPY Curncy"), ("2017-08-11", "DWMFHC Index"),
    ("2022-02-28", "MXRU0EN Index"), ("2000-06-12", "BSETCD Index"), ("2000-09-27", "DWMFCG Index"),
    ("2012-07-03", "BOXBBC9V Index"), ("2021-05-28", "MXGR0FN Index"), ("2013-11-27", "MXGR0FN Index"),
    ("2020-05-11", "DWMFTC Index"), ("2002-07-24", "SASETCSI Index"), ("2002-07-25", "SASETCSI Index"),
    ("2002-07-24", "SASEELE Index"), ("2002-07-25", "SASEELE Index"), ("2006-08-23", "MXAR0IN Index"),
    ("2021-12-01", "MXAR0TC Index"), ("2010-12-02", "MXTH0IN Index"), ("2007-02-27", "DWGFN Index"),
    ("2007-02-28", "DWGFN Index"), ("2021-12-01", "MXAR0FN Index"), ("2013-06-03", "MXMX0HC Index"),
    ("2014-06-02", "MXDK0CD Index"), ("2013-11-27", "MXGR0UT Index"), ("2011-11-16", "JMDEUR Curncy"),
    ("2011-11-17", "JMDEUR Curncy"), ("2010-08-13", "JMDEUR Curncy"), ("2010-08-16", "JMDEUR Curncy"),
    ("2016-03-01", "MXAR0CD Index"), ("2020-06-01", "MXIT0IT Index"), ("2001-12-31", "JFINX Index"),
    ("2000-01-12", "SASETISI Index"), ("2000-01-13", "SASETISI Index"), ("2001-06-20", "SASEELE Index"),
    ("2001-06-21", "SASEELE Index"), ("2013-05-02", "MXGR0TC Index"), ("2021-05-28", "MXRU0CD Index"),
    ("2006-08-23", "MXIT0HC Index"), ("2003-12-11", "FSTUT Index"), ("2022-06-01", "MXAR0CD Index"),
    ("2012-06-01", "MXAT0IN Index"), ("2013-11-27", "MXGR0MT Index"), ("2013-11-27", "MXGR0EN Index"),
    ("2004-09-28", "XUMAL Index"), ("2003-05-29", "F3CONGI Index"), ("2015-12-01", "MXGR0UT Index"),
    ("2016-12-01", "MXAR0UT Index"), ("2002-05-27", "RGUSMS Index"), ("2022-06-01", "MXAR0UT Index"),
    ("2012-12-03", "MXTH0IN Index"), ("2012-01-03", "NGSEFB10 Index"), ("2022-03-11", "JMSMX Index"),
    ("2008-03-26", "MXEG0IN Index"), ("2014-02-18", "NGSEOG5 Index"), ("2017-12-01", "MXPL0IT Index"),
    ("2001-03-19", "DJUSCL Index"), ("2008-04-25", "JMDEUR Curncy"), ("2006-08-23", "MXMX0EN Index"),
    ("2018-09-03", "MXGR0EN Index"), ("2021-11-17", "MXGR0UT Index"), ("2001-01-04", "AEDUSD Curncy"),
    ("2001-01-05", "AEDUSD Curncy"), ("2002-12-06", "AEDUSD Curncy"), ("2002-12-09", "AEDUSD Curncy"),
    ("2005-05-23", "AEDUSD Curncy"), ("2005-05-24", "AEDUSD Curncy"), ("2001-01-01", "DKKEUR Curncy"),
    ("2001-01-02", "DKKEUR Curncy"), ("2000-11-20", "MAGY3YR Index"), ("2000-11-21", "MAGY3YR Index"),
    ("2005-07-21", "CNYUSD Curncy"), ("2022-11-22", "MICXRU20 Index"), ("2022-11-22", "MICXRU10 Index"),
    ("2022-11-22", "MICXRU5Y Index"), ("2022-11-22", "MICXRU3Y Index")
]

for dt, col in aberrant_returns:
    if col in returns.columns:
        try:
            returns.loc[dt, col] = np.nan
        except KeyError:
            pass


slice_aberrants = [
    ('2015-04-14', None, 'BSET Index'),
    ('2008-09-01', None, 'MXRU0UT Index'),
    ('2010-12-02', None, 'MXID0HC Index'),
    ('2016-10-31', '2016-12-21', 'EB03/3M Index'),
    ('2018-03-13', '2018-04-12', 'EB03/3M Index'),
    ('2009-04-07', '2009-07-07', 'NDDR1T Curncy'),
    ('2019-06-06', '2019-06-10', 'BDTEUR Curncy'),
    ('2019-06-06', '2019-06-07', 'BDTUSD Curncy')
]

for start, end, col in slice_aberrants:
    if col in returns.columns:
        try:
            returns.loc[start:end, col] = np.nan
        except KeyError:
            pass

returns.to_csv("230216_returns_19_25.csv")