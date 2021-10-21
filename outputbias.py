

#install stuff in this cell

from google.colab import drive
drive.mount('/content/drive')

#import stuff in this cell
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import scipy
from scipy.stats.mstats import gmean
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
from matplotlib.ticker import FormatStrFormatter

# recommendations_data = pd.read_csv('RecommendationsBX.csv',converters={'item':lambda x:str(x)})
recommendations_data = pd.read_csv('drive/MyDrive/AZDataset/RecommendationsD2.csv',converters={'item':lambda x:str(x)})
print(recommendations_data.head())

recommendations_data_original = recommendations_data.copy(deep=True)
for ind in recommendations_data.index:
  recommendations_data.at[ind,'score'] = recommendations_data['score'][ind] +1.0

recommendations_data_item_item = recommendations_data.loc[recommendations_data['Algorithm'] == 'ItemItem']
recommendations_data_user_user = recommendations_data.loc[recommendations_data['Algorithm'] == 'UserUser']
# recommendations_data_als = recommendations_data.loc[recommendations_data['Algorithm'] == 'Als']
print(recommendations_data_item_item.head())
print(recommendations_data_user_user.head())

author_credentials = pd.read_csv('drive/MyDrive/AZDataset/genders.csv',sep='@',converters={'isbn':lambda x: str(x)})
print(author_credentials.head())

#unique_users = pd.read_csv('BX_Unique_Users.csv',converters={'userid':lambda x: str(x)})
unique_users_item_item = recommendations_data_item_item['user'].unique()
#print(len(unique_users))
unique_users_list_item_item = list(unique_users_item_item)
print(len(unique_users_list_item_item))
unique_users_user_user = recommendations_data_user_user['user'].unique()
unique_users_list_user_user = list(unique_users_user_user)
print(len(unique_users_list_user_user))
#print(len(unique_users_list))
# unique_users_als = recommendations_data_als['user'].unique()
# unique_users_list_als = list(unique_users_als)
# print(len(unique_users_list_als))

df_of_interest = recommendations_data_item_item
user_list_of_interest = unique_users_list_item_item

def lookup_gender(isbn):
  author_details = author_credentials.loc[author_credentials['isbn'] == isbn]
  if author_details.empty:
    return None
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

def judge_recommendation_data_segment(data_segment):
  male_author_books_score = []
  female_author_books_score = []
  # male_author_books_score = 0.0
  # female_author_books_score = 0.0
  for ind in data_segment.index:
    gender = lookup_gender(data_segment['item'][ind])
    if gender == 'male':
      male_author_books_score.append(data_segment['score'][ind])
      #male_author_books_score = male_author_books_score + 1.0
    elif gender == 'female':
      female_author_books_score.append(data_segment['score'][ind])
      #female_author_books_score = female_author_books_score + 1.0
    else:
      print('Gender identification failed for item: ',data_segment['item'][ind])
  if not male_author_books_score:
    return len(male_author_books_score),len(female_author_books_score),np.nan,np.nan
  if not female_author_books_score:
    return len(male_author_books_score),len(female_author_books_score),np.nan,np.nan
  # average_score_male_author_books = sum(male_author_books_score)/len(male_author_books_score)
  # average_score_female_author_books = sum(female_author_books_score)/len(female_author_books_score)
  # average_score_male_author_books = average_score_male_author_books #+ 1.0
  # average_score_female_author_books = average_score_female_author_books #+ 1.0
  gmean_rating_male_author_books = GeometricMean(male_author_books_score)
  gmean_rating_female_author_books = GeometricMean(female_author_books_score)
  # ratio_male_female = average_score_male_author_books/average_score_female_author_books
  # male_author_books_score = male_author_books_score + 1
  # female_author_books_score = female_author_books_score + 1
  # ratio_male_female = male_author_books_score +1.0 / female_author_books_score +1.0
  if gmean_rating_male_author_books == -1:
    return len(male_author_books_ratings),len(female_author_books_ratings),np.nan,np.nan
  if gmean_rating_female_author_books == -1:
    return len(male_author_books_ratings),len(female_author_books_ratings),np.nan,np.nan  
  ratio_male_female = gmean_rating_male_author_books / gmean_rating_female_author_books
  bias_score = np.log(ratio_male_female)
  #return len(male_author_books_score),len(female_author_books_score),ratio_male_female,bias_score
  #print('male score: ',male_author_books_score, 'female score: ', female_author_books_score)
  return len(male_author_books_score),len(female_author_books_score),ratio_male_female,bias_score

list_recommendation_bias_score = []
strToWriteList = []
i = 0
#for item item evaluation
for user in user_list_of_interest:
  if i%100 == 0:
    print(i)
  user_data_segment = df_of_interest.loc[df_of_interest['user'] == user]
  user_data_segment_filtered = user_data_segment[['item','score']]
  no_male_author_books,no_female_author_books,ratio_male_female,user_bias_score = judge_recommendation_data_segment(user_data_segment_filtered)
  strToWrite = '\n'+str(user)+','+str(no_male_author_books)+','+str(no_female_author_books)+','+str(ratio_male_female)+','+str(user_bias_score)
  strToWriteList.append(strToWrite)
  list_recommendation_bias_score.append(user_bias_score)
  i+=1
  # print(strToWrite)

recommendationsBiasFile = open('drive/MyDrive/BXDataset/modelRecommendationsBias_ItemItem.csv','w')
recommendationsBiasFile.write('user,no_male_author_books,no_female_author_books,ratio_male_female,user_bias_score')
for item in strToWriteList:
  recommendationsBiasFile.write(item)
recommendationsBiasFile.close()

print(list_recommendation_bias_score)
cleaned_list_recommendation_bias_score = [x for x in list_recommendation_bias_score if str(x) != 'nan']
print(cleaned_list_recommendation_bias_score)
mean,std=norm.fit(cleaned_list_recommendation_bias_score)
print(mean)
print(std)

fig, ax = matplotlib.pyplot.subplots(figsize=(4, 3), dpi=(200))
# matplotlib.pyplot.figure(figsize=(4,3))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
matplotlib.pyplot.hist(cleaned_list_recommendation_bias_score,density=True, bins=100)
# xmin, xmax = plt.xlim()
# print(xmin)
# print(xmax)
x = np.linspace(-1.5, 1.5, 1000)
y = norm.pdf(x, mean, std)
matplotlib.pyplot.xlabel('Log bias', fontsize=12)
matplotlib.pyplot.ylabel('Frequency density',fontsize=12)
matplotlib.pyplot.plot(x, y)
matplotlib.pyplot.show()

# user_input_bias = pd.read_csv('drive/MyDrive/AZDataset/UserBiasD2.csv', sep='@')
# print(user_input_bias.head())
# user_input_bias = user_input_bias[['user','bias']]
# user_input_bias = user_input_bias[user_input_bias['bias'].notna()]
# print(user_input_bias.head())

# user_recommendation_bias = pd.read_csv('recommendationsBias_Als.csv')
# print(user_recommendation_bias.head())
# user_recommendation_bias = user_recommendation_bias[['user','user_bias_score']]
# user_recommendation_bias = user_recommendation_bias[user_recommendation_bias['user_bias_score'].notna()]
# print(user_recommendation_bias.head())

# input_bias = []

# output_bias = []

# for user,score in zip(user_recommendation_bias['user'],user_recommendation_bias['user_bias_score']):
#   temp_score = user_input_bias[user_input_bias['user'] == user].iloc[0:1,1:2]['bias']
#   #print(type(temp_score))
#   if not temp_score.empty:
#     input_bias.append(float(list(temp_score)[0]))
#     output_bias.append(score)
#     #print('User',str(user),'score',score)

# print(len(input_bias))

# print(len(output_bias))

# df_input_bias = pd.DataFrame(input_bias,columns=['bias_score'])
# print(df_input_bias.head())
# df_output_bias = pd.DataFrame(output_bias,columns=['bias_score'])
# print(df_output_bias.head())

# input_bias = np.array(input_bias)
# output_bias = np.array(output_bias)
# input_bias.reshape((-1, 1))
# print(input_bias)
# print('next')
# print(output_bias)

# regr = LinearRegression()
# regr.fit(df_input_bias, df_output_bias)
# print('Slope: \n', regr.coef_)
# print('Intercept: ', regr.intercept_)
# slope = regr.coef_[0]
# intercept = regr.intercept_[0]

# fig=plt.figure()
# ax=fig.add_axes([0,0,1,1])
# ax.scatter(df_input_bias,df_output_bias,color='r',s=1)
# x_vals = np.array(ax.get_xlim())
# y_vals = intercept + slope * x_vals
# ax.plot(x_vals, y_vals, '--')
# ax.axhline(0, color='black')
# ax.axvline(0, color='black')
# ax.set_xlabel('Input bias')
# ax.set_ylabel('Output bias')
# ax.set_title('Output bias vs Input bias')

# # plt.scatter(df_input_bias, df_output_bias,  color='black',s = 1)
# # # plt.plot(df_input_bias, df_output_bias, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())
# def abline(slope, intercept):
#     """Plot a line from slope and intercept"""
#     axes = plt.gca()
#     x_vals = np.array(axes.get_xlim())
#     y_vals = intercept + slope * x_vals
#     plt.plot(x_vals, y_vals, '--')
# abline(regr.coef_[0],regr.intercept_[0])
# plt.show()

