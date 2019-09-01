



#%%
import pandas as pd
import os
import matplotlib as plot
os.chdir(os.path.join(os.getcwd(), 'House Price Regression/data'))
df = pd.read_csv("train.csv")
    

#%%

import missingno
missingno.matrix(df, figsize=(len(df.columns),10))

#%%
missingno.matrix(df, figsize=(30,10))

#%%
missingno.matrix(df, figsize=(len(df.columns),20))

#%%
