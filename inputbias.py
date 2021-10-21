
#install stuff in this cell

from google.colab import drive
drive.mount('/content/drive')

#import stuff in this cell
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib
from scipy.stats.mstats import gmean
from scipy.stats.mstats import gmean
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
from matplotlib.ticker import FormatStrFormatter

# rating_dataset = pd.read_csv('drive/MyDrive/BXDataset/ratingsBiased.csv',sep='@',converters={'item':lambda x:str(x)})
rating_dataset = pd.read_csv('drive/MyDrive/AZDataset/ratings2.csv',sep='@',converters={'item':lambda x:str(x)})
# rating_dataset = rating_dataset.drop('index', axis=1)
# for ind in rating_dataset.index:
#   rating_dataset.at[ind,'item'] = rating_dataset['item'][ind] + '##'
print(rating_dataset.head())
print(rating_dataset.shape)
# print(rating_dataset[rating_dataset['rating'] == 0].head())

for ind in rating_dataset.index:
  rating_dataset.at[ind,'rating'] = rating_dataset['rating'][ind] +1.0

# print(rating_dataset.head())

author_credentials = pd.read_csv('drive/MyDrive/AZDataset/genders.csv'\
                                 ,sep='@',converters={'isbn':lambda x: str(x)})
print(author_credentials.head())
# for ind2 in author_credentials.index:
#   author_credentials.at[ind,'isbn'] = author_credentials['isbn'][ind] + '##'

# unique_users = pd.read_csv('BX_Unique_Users.csv',converters={'userid':lambda x: str(x)})
# print(unique_users.head())
# unique_users_list = unique_users['userid'].tolist()
# print(unique_users_list)
unique_users = rating_dataset['user'].unique()
unique_users_list = list(unique_users)
print(len(unique_users_list))

def lookup_gender(isbn):
  author_details = author_credentials.loc[author_credentials['isbn'] == isbn]
  if author_details.empty:
    # print(f"None --- {isbn}")
    return None
  # print(f"{list(author_details['gender'])[0]} --- {isbn}")
  return list(author_details['gender'])[0]

def GeometricMean(number_list):
  if len(number_list) == 0:
    return -1
  return gmean(number_list)
  # product = 1.0
  # for num in number_list:
  #   product = product*num
  # gm = product ** (1/len(number_list))
  # return gm

def judge_user_data_segment(data_segment):
  male_author_books_ratings = []
  female_author_books_ratings = []
  for ind in data_segment.index:
    gender = lookup_gender(data_segment['item'][ind])
    if gender == 'male':
      male_author_books_ratings.append(data_segment['rating'][ind])
    elif gender == 'female':
      female_author_books_ratings.append(data_segment['rating'][ind])
    else:
      return np.nan,np.nan,np.nan,np.nan
      print('Gender identification failed for isbn: ',data_segment['item'][ind])
  if not male_author_books_ratings:
    return len(male_author_books_ratings),len(female_author_books_ratings),np.nan,np.nan
  if not female_author_books_ratings:
    return len(male_author_books_ratings),len(female_author_books_ratings),np.nan,np.nan
  # average_rating_male_author_books = sum(male_author_books_ratings)/len(male_author_books_ratings)
  # average_rating_female_author_books = sum(female_author_books_ratings)/len(female_author_books_ratings)
  # sum_rating_male_author_books = sum(male_author_books_ratings) + 0.01
  # sum_rating_female_author_books = sum(female_author_books_ratings) + 1.0
  gmean_rating_male_author_books = GeometricMean(male_author_books_ratings)
  gmean_rating_female_author_books = GeometricMean(female_author_books_ratings)
  if gmean_rating_male_author_books == -1:
    return len(male_author_books_ratings),len(female_author_books_ratings),np.nan,np.nan
  if gmean_rating_female_author_books == -1:
    return len(male_author_books_ratings),len(female_author_books_ratings),np.nan,np.nan
  ratio_male_female = gmean_rating_male_author_books/gmean_rating_female_author_books
  bias_score = np.log(ratio_male_female)
  # bias_score = ratio_male_female
  return len(male_author_books_ratings),len(female_author_books_ratings),ratio_male_female,bias_score

list_user_bias_score = []
strToWriteList = []

i = 0
for user in unique_users_list:
  if i%100 == 0:
    print(i)
  # user_data_segment = rating_dataset.loc[rating_dataset['user'] == int(user)]
  user_data_segment = rating_dataset.loc[rating_dataset['user'] == user]
  user_data_segment_filtered = user_data_segment[['item','rating']]
  no_male_author_books,no_female_author_books,ratio_male_female,user_bias_score = judge_user_data_segment(user_data_segment_filtered)
  strToWrite = '\n'+str(user)+'@'+str(no_male_author_books)+'@'+str(no_female_author_books)+'@'+str(ratio_male_female)+'@'+str(user_bias_score)
  # userBiasFile.write(strToWrite)
  strToWriteList.append(strToWrite)
  list_user_bias_score.append(user_bias_score)
  i += 1
  #print(strToWrite)
# userBiasFile.close()

print(len(unique_users_list))
print(len(list_user_bias_score))

# userBiasFile = open('drive/MyDrive/AZDataset/UserBiasD2.csv','w')
userBiasFile = open('drive/MyDrive/AZDataset/inputUserBiasD1.csv','w')
for item in zip(unique_users_list, list_user_bias_score):
  strToWrite = '\n' +str(item[0]) + '@' + str(item[1])
  # print(strToWrite) 
  userBiasFile.write(strToWrite)

cleaned_list_user_bias_score = [x for x in list_user_bias_score if str(x) != 'nan' and str(x) != '-inf' and str(x) != 'inf']
# cleaned_list_user_bias_score = [x for x in list_user_bias_score if (x > 0.08 or (x < 0 and x > -0.005)) and str(x) != 'nan' and str(x) != '-inf' and str(x) != 'inf'] # cleaned_list_user_bias_score = [x for x in list_user_bias_score if str(x) != 'nan' and str(x) != '-inf' and str(x) != 'inf']
# print(max(cleaned_list_user_bias_score))
print(len(cleaned_list_user_bias_score))
mean,std=norm.fit(cleaned_list_user_bias_score)
print(mean)
print(std)

fig, ax = matplotlib.pyplot.subplots(dpi=(200))
# matplotlib.pyplot.figure(figsize=(4,3))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
matplotlib.pyplot.hist(cleaned_list_user_bias_score,density=True, bins=100)
# xmin, xmax = plt.xlim()
# print(xmin)
# print(xmax)
x = np.linspace(-1.5,1.5, 1000)
y = norm.pdf(x, mean, std)
matplotlib.pyplot.xlabel('Log bias', fontsize=12)
matplotlib.pyplot.ylabel('Frequency density',fontsize=12)
matplotlib.pyplot.plot(x, y)
matplotlib.pyplot.show()

