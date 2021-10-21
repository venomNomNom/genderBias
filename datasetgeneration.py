

import pandas as pd
import numpy as np
dfbx = pd.read_csv ('filebx.csv', sep=";", converters={'ISBN': lambda x: str(x)})
print(dfbx.head())

uniqueISBNbx = dfbx["ISBN"].unique()
print(uniqueISBNbx)

dfaz = pd.read_csv ('fileaz.csv', converters={'item': lambda x: str(x)})
print(dfaz.head())

uniqueISBNaz = dfaz["item"].unique()
print(type(uniqueISBNaz))

uniqueISBNaz.tofile('uniqueISBNaz.csv', sep="\n", format="%s")
uniqueISBNbx.tofile('uniqueISBNbx.csv', sep="\n", format="%s")

print('abc')



