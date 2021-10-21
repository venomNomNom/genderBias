

import pandas as pd
import numpy as np
from collections import Counter

#dfaz = pd.read_csv ('fileaz.csv', converters={'item': lambda x: str(x)})
dfbx = pd.read_csv ('filebx.csv', sep=";", converters={'ISBN': lambda x: str(x)})

#dfaz = dfaz[dfaz['rating'] > 0]
dfbx = dfbx[dfbx['Book-Rating'] > 0]

#isbnaz = dfaz['item']
isbnbx = dfbx['ISBN']

#print(len(isbnaz))
print(len(isbnbx))

def removeElements(lst, k): 
    counted = Counter(lst) 
    return [el for el in lst if counted[el] >= k]

#isbnaz = removeElements(isbnaz, 400)
isbnbx = removeElements(isbnbx, 25)

#print(len(isbnaz))
print(len(isbnbx))

#dfaz_filtered_books = dfaz[dfaz['item'].isin(isbnaz)]
dfbx_filtered_books = dfbx[dfbx['ISBN'].isin(isbnbx)]

#print(len(dfaz_filtered_books))
print(len(dfbx_filtered_books))

#useraz = dfaz_filtered_books['user']
userbx = dfbx_filtered_books['User-ID']

#print(len(useraz))
print(len(userbx))

#useraz = removeElements(useraz, 400)
userbx = removeElements(userbx, 25)

#print(len(useraz))
print(len(userbx))

#dfaz_filtered_books_users = dfaz_filtered_books[dfaz_filtered_books['user'].isin(useraz)]
dfbx_filtered_books_users = dfbx_filtered_books[dfbx_filtered_books['User-ID'].isin(userbx)]

# dfaz_filtered_books_users['item'] = dfaz_filtered_books_users['item'].astype('str')
# dfbx_filtered_books_users['ISBN'] = dfbx_filtered_books_users['ISBN'].astype('str')

#dfaz_filtered_books_users.to_csv("pruned_dataset_az.csv", sep=",")
dfbx_filtered_books_users.to_csv("pruned_dataset_bx.csv", sep=",")

