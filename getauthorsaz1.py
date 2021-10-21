

import pandas as pd
import numpy as np

dfisbnaz = pd.read_csv('failedISBNAuthors_1.csv')
print(dfisbnaz.head())

print(type(dfisbnaz['isbn'][0]))

pip install ratelimit

import requests
import json
import time
from ratelimit import limits

@limits(calls=1, period=1)
def writeBookDetails(isbn, fileptr1, fileptr2):
  URL = "https://www.googleapis.com/books/v1/volumes?q=isbn:" + isbn
  response = requests.get(url = URL)
  results = response.json()
  try:
    if (results['totalItems'] != 0):
      book = results['items'][0]
      if "authors" in book["volumeInfo"]:
        authors = (book["volumeInfo"]["authors"])
        firstAuthor = authors[0]
        authorFirstName = firstAuthor.split(' ', 1)[0]
        gender = 'male'
        strToWrite = '\n' + isbn + ',' + authors[0]
        fileptr1.write(strToWrite)
        print(strToWrite)
      else:
        strToWrite = '\n' + isbn
        fileptr2.write(strToWrite)
        print("author key does not exists")
        print(strToWrite)
    else:
      strToWrite = '\n' + isbn
      fileptr2.write(strToWrite)
      print("book not in database")
      print(strToWrite)
  except:
      strToWrite = '\n' + isbn
      fileptr2.write(strToWrite)
      print("error occured")
      print(isbn)

f_isbn_author_AZ = open("recoveredISBNAuthors_1.csv", "w")
failedAttemptsAZ = open("reFailedISBNAuthors_1.csv", "w")
for isbnNo in dfisbnaz['isbn']:
  writeBookDetails(isbnNo, f_isbn_author_AZ, failedAttemptsAZ)
  time.sleep(1)
f_isbn_author_AZ.close()
failedAttemptsAZ.close()

