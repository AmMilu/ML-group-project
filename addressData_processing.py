#read data
import numpy as np
import pandas as pd
df = pd.read_csv('clean_data.csv', header=0)
x = df.groupby('address')
orginial_address = list(x.groups.keys())
address_price = x.mean().sort_values(by=["price"])
address_price.to_csv("tmp.csv")
new_df = pd.read_csv('tmp.csv',header=0)
new_address = new_df.iloc[:,0]

df["address"] = df["address"].replace(orginial_address,new_address)
df.to_csv("updated_data.csv",index=False)