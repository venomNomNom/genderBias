

pip install lenskit

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als,basic,item_knn as iknn, user_knn as uknn
from scipy.stats import norm
import numpy as np
from scipy.stats.mstats import gmean
import math
from lenskit.batch import predict
from lenskit.metrics.predict import rmse, mae

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

rating_dataset = pd.read_csv('drive/MyDrive/BXDataset/ratingsBiased.csv',sep='@',converters={'isbn':lambda x:str(x)})
# rating_dataset = rating_dataset.drop('index',axis=1)
print(rating_dataset.head())

for ind in rating_dataset.index:
  rating_dataset.at[ind,'rating'] = rating_dataset['rating'][ind] +1.0

author_credentials = pd.read_csv('drive/MyDrive/BXDataset/genders.csv',sep=',',converters={'isbn':lambda x: str(x)})
print(author_credentials.head())

algo_ii = iknn.ItemItem(20)
algo_uu = uknn.UserUser(20)
algo_als = als.BiasedMF(50)
algo_pop = basic.Popular()

def eval(aname, algo, train, test):
  fittable = util.clone(algo)
  fittable = Recommender.adapt(fittable)
  fittable.fit(train)
  users = test.user.unique()
  # now we run the recommender
  recs = batch.recommend(fittable, users, 16)
  # add the algorithm name for analyzability
  recs['Algorithm'] = aname
  return recs

def lookup_gender(isbn):
  author_details = author_credentials.loc[author_credentials['isbn'] == isbn]
  if author_details.empty:
    return 'None'
  return list(author_details['gender'])[0]

def GeometricMean(number_list):
  if len(number_list) == 0:
    return -1
  return gmean(number_list)

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
      print('Gender identification failed for isbn: ',data_segment['item'][ind])
  if not male_author_books_ratings:
    return len(male_author_books_ratings),len(female_author_books_ratings),np.nan,np.nan
  if not female_author_books_ratings:
    return len(male_author_books_ratings),len(female_author_books_ratings),np.nan,np.nan
  # average_rating_male_author_books = sum(male_author_books_ratings)/len(male_author_books_ratings)
  # average_rating_female_author_books = sum(female_author_books_ratings)/len(female_author_books_ratings)
  gmean_rating_male_author_books = GeometricMean(male_author_books_ratings)
  gmean_rating_female_author_books = GeometricMean(female_author_books_ratings)
  if gmean_rating_male_author_books == -1:
    return len(male_author_books_ratings),len(female_author_books_ratings),np.nan,np.nan
  if gmean_rating_female_author_books == -1:
    return len(male_author_books_ratings),len(female_author_books_ratings),np.nan,np.nan
  #sum_rating_male_author_books = sum(male_author_books_ratings) + 1.0
  #sum_rating_female_author_books = sum(female_author_books_ratings) + 1.0
  ratio_male_female = gmean_rating_male_author_books/gmean_rating_female_author_books
  bias_score = np.log(ratio_male_female)
  return len(male_author_books_ratings),len(female_author_books_ratings),ratio_male_female,bias_score

def getGlobalBias(train):
  unique_users = train['user'].unique()
  unique_users_list = list(unique_users)
  list_user_bias_score = []
  for user in unique_users_list:
    user_data_segment = train.loc[train['user'] == user]
    user_data_segment_filtered = user_data_segment[['item','rating']]
    no_male_author_books,no_female_author_books,ratio_male_female,user_bias_score = judge_user_data_segment(user_data_segment_filtered)
    #strToWrite = '\n'+str(user)+','+str(no_male_author_books)+','+str(no_female_author_books)+','+str(ratio_male_female)+','+str(user_bias_score)
    #userBiasFile.write(strToWrite)
    list_user_bias_score.append(user_bias_score)
    #print(strToWrite)
  #userBiasFile.close()
  cleaned_list_user_bias_score = [x for x in list_user_bias_score if str(x) != 'nan' and str(x) != '-inf' and str(x) != 'inf']
  mean,std=norm.fit(cleaned_list_user_bias_score)
  print(mean)
  return mean

def findGmBookRating(train, book):
  filtered_data = train[train['item'] == book]
  ratings = list(filtered_data['rating'])
  for i in range(0,len(ratings)):
    ratings[i] = int(ratings[i])
  gm_rating = GeometricMean(ratings)
  return gm_rating

def calculateUserBiases(train, global_bias):
  unique_users = train['user'].unique()
  unique_users_list = list(unique_users)
  user_id = []
  book_bias_scores = []
  book_isbn = []
  for user in unique_users_list:
    user_data_segment = train.loc[train['user'] == user]
    user_data_segment_filtered = user_data_segment[['item','rating']]
    user_data_segment_filtered['authorGender'] = user_data_segment_filtered.apply(lambda x : lookup_gender(x['item']), axis=1)
    female_books_user_data_segment_filtered = user_data_segment_filtered.loc[user_data_segment_filtered['authorGender'] == 'female']
    for book in list(female_books_user_data_segment_filtered['item'].unique()):
      gm_book_rating = findGmBookRating(train, book)
      adjusted_gm_book_rating = gm_book_rating*math.exp(global_bias)
      book_df = female_books_user_data_segment_filtered.loc[female_books_user_data_segment_filtered['item'] == book]
      book_rating = book_df['rating']
      user_bias = adjusted_gm_book_rating / list(book_rating)[0]
      book_bias_scores.append(user_bias)
      book_isbn.append(book)
      user_id.append(user)
    average_user_bias_score = 1
    # if len(book_bias_scores) != 0:
    #   average_user_bias_score = sum(book_bias_scores)/len(book_bias_scores)
    # user_bias_scores.append(average_user_bias_score)
  data_tuples = list(zip(user_id,book_isbn,book_bias_scores))
  return pd.DataFrame(data_tuples, columns=['user','isbn','bias'])

def adjustDataset(train, user_biases):
  print(train)
  unique_users = train['user'].unique()
  unique_users_list = list(unique_users)
  for user in unique_users_list:
    filter_user_rows = user_biases.loc[user_biases['user'] == user]
    for ind in train.index:
      if train['user'][ind] == user:
        if lookup_gender(train['item'][ind]) == 'female':
          bias_shown = list(filter_user_rows.loc[filter_user_rows['isbn'] == train['item'][ind]]['bias'])[0]
          train.at[ind,'rating'] = train['rating'][ind] * bias_shown
          # print(bias_shown)
  return train

def process(train):
  # global_bias = getGlobalBias(train)
  global_bias = 0.2436
  user_biases = calculateUserBiases(train, global_bias)
  adjusted_train_dataset = adjustDataset(train, user_biases)
  return adjusted_train_dataset,user_biases

def readjust(recommendations, user_biases, input_user_bias):
  unique_users = recommendations['user'].unique()
  unique_users_list = list(unique_users)
  for user in unique_users_list:
    filter_user_row = user_biases.loc[user_biases['user'] == user]
    user_bias = list(filter_user_row['bias'])
    avg_user_bias = sum(user_bias)/len(user_bias)
    # user_row = input_user_bias.loc[input_user_bias['user'] == user]
    # avg_user_bias = list(user_row['bias'])[0]
    #if user_bias > 1.0:
    for ind in recommendations.index:
      if recommendations['user'][ind] == user:
        if lookup_gender(recommendations['item'][ind]) == 'female':
          recommendations.at[ind,'score'] = recommendations['score'][ind] / math.exp(avg_user_bias)
          #print('done')
  return recommendations

def readjustPredictions(predictions, user_biases):
  unique_users = predictions['user'].unique()
  unique_users_list = list(unique_users)
  for user in unique_users_list:
    filter_user_row = user_biases.loc[user_biases['user'] == user]
    user_bias = list(filter_user_row['bias'])
    if len(user_bias) == 0:
      continue
    avg_user_bias = sum(user_bias)/len(user_bias)
    # user_row = input_user_bias.loc[input_user_bias['user'] == user]
    # avg_user_bias = list(user_row['bias'])[0]
    #if user_bias > 1.0:
    for ind in predictions.index:
      if predictions['user'][ind] == user:
        if lookup_gender(predictions['item'][ind]) == 'female':
          predictions.at[ind,'prediction'] = predictions['prediction'][ind] / math.exp(avg_user_bias)
          #print('done')
  return predictions

all_recs = []
test_data = []
user_rmse_ii_list = []
user_rmse_uu_list = []
user_mae_ii_list = []
user_mae_uu_list = []
# input_user_bias = pd.read_csv('userBiasFile')
for train, test in xf.partition_users(rating_dataset[['user', 'item', 'rating']], 1, xf.SampleFrac(0.2)):
    test_data.append(test)
    adjusted_train, user_biases = process(train)
    # print(adjusted_train.head())
    #adjusted_train_2, user_biases_2 = process(adjusted_train)
    #print(adjusted_train.head())
    item_item_recs = eval('ItemItem', algo_ii, adjusted_train, test)
    disadjusted_item_item_recs = readjust(item_item_recs, user_biases)
    # #disadjusted_item_item_recs_2 = readjust(disadjusted_item_item_recs, user_biases)
    user_user_recs = eval('UserUser', algo_uu, adjusted_train, test)
    disadjusted_user_user_recs = readjust(user_user_recs, user_biases)
    # item_item_recs = disadjusted_item_item_recs #OPEN FOR ACCURACY
    # user_user_recs = diadjusted_user_user_recs  #OPEN FOR ACCURACY
    #disadjusted_user_user_recs_2 = readjust(disadjusted_user_user_recs, user_biases)
    #pop_recs = eval('Pop', algo_pop, train, test)
    algo_ii_2 = iknn.ItemItem(20)
    algo_uu_2 = uknn.UserUser(20)
    algo_ii_2.fit(adjusted_train)
    algo_uu_2.fit(adjusted_train)
    preds_ii = predict(algo_ii_2, test)
    preds_uu = predict(algo_uu_2, test)
    # print(preds_ii.head())
    disadjusted_preds_ii = readjustPredictions(preds_ii, user_biases)
    disadjusted_preds_uu = readjustPredictions(preds_uu, user_biases)
    # preds_ii = disadjusted_preds_ii   #OPEN FOR ACCURACY
    # preds_uu = disadjusted_preds_uu   #OPEN FOR ACCURACY
    user_rmse_ii = preds_ii.groupby('user').apply(lambda df: rmse(df.prediction, df.rating))
    user_rmse_ii_list.append(user_rmse_ii.mean())
    user_rmse_uu = preds_uu.groupby('user').apply(lambda df: rmse(df.prediction, df.rating))
    user_rmse_uu_list.append(user_rmse_uu.mean())
    user_mae_ii = preds_ii.groupby('user').apply(lambda df: mae(df.prediction, df.rating))
    user_mae_ii_list.append(user_rmse_ii.mean())
    user_mae_uu = preds_uu.groupby('user').apply(lambda df: mae(df.prediction, df.rating))
    user_mae_uu_list.append(user_rmse_uu.mean())   
    all_recs.append(item_item_recs)
    all_recs.append(user_user_recs)
    print('new epoch')
    #all_recs.append(pop_recs)

all_recs = pd.concat(all_recs, ignore_index=True)
all_recs.head()

all_recs.to_csv('ModelRecommendations_with_accuracy.csv')

test_data = pd.concat(test_data, ignore_index=True)

rla = topn.RecListAnalysis()
rla.add_metric(topn.ndcg)
rla.add_metric(topn.precision)
rla.add_metric(topn.recall)
results = rla.compute(all_recs, test_data2)
results.head()

results.groupby('Algorithm').ndcg.mean()

results.groupby('Algorithm').precision.mean()

results.groupby('Algorithm').recall.mean()

results.groupby('Algorithm').ndcg.mean().plot.bar()

results.groupby('Algorithm').precision.mean().plot.bar()

results.groupby('Algorithm').recall.mean().plot.bar()
