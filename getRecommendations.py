

pip install lenskit

pip install --upgrade tbb

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als,basic,item_knn as iknn, user_knn as uknn
from lenskit.batch import predict
from lenskit.metrics.predict import rmse, mae

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

rating_dataset = pd.read_csv('drive/MyDrive/AZDataset/ratings2.csv', sep='@')
print(rating_dataset.head())

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
    recs = batch.recommend(fittable, users, 20)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs

all_recs = []
test_data = []
user_rmse_ii_list = []
user_rmse_uu_list = []
user_mae_ii_list = []
user_mae_uu_list = []
for train, test in xf.partition_users(rating_dataset[['user', 'item', 'rating']], 1, xf.SampleFrac(0.2)):
    print("new epoch")
    test_data.append(test)
    # als_recs = eval('Als', algo_als, train, test)
    # item_item_recs = eval('ItemItem', algo_ii, train, test)
    # user_user_recs = eval('UserUser', algo_uu, train, test)
    # pop_recs = eval('Pop', algo_pop, train, test)     
    algo_ii_2 = iknn.ItemItem(50)
    algo_uu_2 = uknn.UserUser(50)
    algo_ii_2.fit(train)
    algo_uu_2.fit(train)
    preds_ii = predict(algo_ii_2, test)
    preds_uu = predict(algo_uu_2, test)
    user_rmse_ii = preds_ii.groupby('user').apply(lambda df: rmse(df.prediction, df.rating))
    user_rmse_ii_list.append(user_rmse_ii.mean())
    user_rmse_uu = preds_uu.groupby('user').apply(lambda df: rmse(df.prediction, df.rating))
    user_rmse_uu_list.append(user_rmse_uu.mean())
    user_mae_ii = preds_ii.groupby('user').apply(lambda df: mae(df.prediction, df.rating))
    user_mae_ii_list.append(user_mae_ii.mean())
    user_mae_uu = preds_uu.groupby('user').apply(lambda df: mae(df.prediction, df.rating))
    user_mae_uu_list.append(user_mae_uu.mean())
    # all_recs.append(item_item_recs)
    # all_recs.append(user_user_recs)
    # all_recs.append(pop_recs)
    # all_recs.append(als_recs)

all_recs = pd.concat(all_recs, ignore_index=True)
# all_recs = all_recs.loc[~all_recs.index.duplicated(keep='first')]
all_recs.head()

all_recs.to_csv('RecommendationsD1.csv')

test_data = pd.concat(test_data, ignore_index=True)
# test_data = test_data.loc[~test_data.index.duplicated(keep='first')]

rla = topn.RecListAnalysis()
rla.add_metric(topn.ndcg)
rla.add_metric(topn.precision)
rla.add_metric(topn.recall)
results = rla.compute(all_recs, test_data)
results.head()

results.groupby('Algorithm').ndcg.mean()

results.groupby('Algorithm').precision.mean()

results.groupby('Algorithm').recall.mean()

results.groupby('Algorithm').ndcg.mean().plot.bar()

results.groupby('Algorithm').precision.mean().plot.bar()

results.groupby('Algorithm').recall.mean().plot.bar()

avg_user_rmse_ii = sum(user_rmse_ii_list)/len(user_rmse_ii_list)
avg_user_rmse_uu = sum(user_rmse_uu_list)/len(user_rmse_uu_list)
avg_user_mae_ii = sum(user_mae_ii_list)/len(user_mae_ii_list)
avg_user_mae_uu = sum(user_mae_uu_list)/len(user_mae_uu_list)

print(f"average user rmse_item_item is {avg_user_rmse_ii}")
print(f"average user rmse_user_user is {avg_user_rmse_uu}")
print(f"average user mae_item_item is {avg_user_mae_ii}")
print(f"average user mae_user_user is {avg_user_mae_uu}")
