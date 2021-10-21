
import pandas as pd

dfGender = pd.read_csv("gender.csv",converters={'ISBN':lambda x:str(x)})
print(dfGender.head())

dfIsbn = pd.read_csv("isbn.csv",converters={'ISBN':lambda x:str(x)})
print(dfIsbn.head())

isbn = list(dfIsbn['isbn'])
genderIsbn = list(dfGender['isbn'])

print(isbn)
print(genderIsbn)

for isbnNo in isbn:
  if isbnNo not in genderIsbn:
    print(isbnNo)
