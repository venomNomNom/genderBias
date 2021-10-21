

pip install ratelimit

import pandas as pd
import requests
import json
import time
from ratelimit import limits
import itertools

df = pd.read_csv ('uniqueNameList.csv')
print(df.head())

df_name = list(df['name'])

@limits(calls=1, period=1)
def getGender(name, genderFile, failureFile):
  name2 = name.split(' ', 1)[0]
  #https://api.genderize.io?name=peter&apikey={YOUR_API_KEY}
  URL = "https://api.genderize.io?name=" + name2+'&apikey=0c98e66f7185f4cf79a05c8b8b2c4c70'
  response = requests.get(url = URL)
  results = response.json()
  try:
    if (results['probability'] > 0.9):
      strToWrite = '\n' + name + '@' + results['gender']
      genderFile.write(strToWrite)
      print(strToWrite)
    else:
      strToWrite = '\n' + name
      failureFile.write(strToWrite)
      print('probability too low')
      print(strToWrite)
  except:
      strToWrite = '\n' + name
      failureFile.write(strToWrite)
      print("Error occured")
      print(strToWrite)

f_name_gender = open("nameGenders.csv", "w")
f_gender_not_found = open("genderNotFound.csv", "w")
for name in df_name:
  getGender(name, f_name_gender, f_gender_not_found)
  time.sleep(1)
f_name_gender.close()
f_gender_not_found.close()
