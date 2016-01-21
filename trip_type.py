__author__ = 'Raihan Masud'
#todo: file needs cleanup
##https://www.kaggle.com/c/walmart-recruiting-trip-type-classification
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../xgboost/wrapper')
import xgboost as xgb

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_selection import chi2
from sklearn import preprocessing

pd.options.mode.chained_assignment = None
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.style.use('ggplot')
from sklearn.metrics import accuracy_score
import pylab as pl
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.externals import joblib
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
import pickle
import sys
from sklearn.grid_search import GridSearchCV
from joblib import Parallel, delayed
import time

'''
ScanCount: how many items purchased, -ve means return
TripType_999 is an "other" category

TripType - a categorical id representing the type of shopping trip the customer made.
This is the ground truth that you are predicting. TripType_999 is an "other" category.

VisitNumber - an id corresponding to a single trip by a single customer

Weekday - the weekday of the trip

Upc - the UPC number of the product purchased

ScanCount - the number of the given item that was purchased. A negative value indicates a product return.

DepartmentDescription - a high-level description of the item's department

FinelineNumber - a more refined category for each of the products, created by Walmart

'''

'''
problem representation:
unique visit on weekday or weekend, someone visited d number of departments(d1, d2, ..) and
more specifically finer_line(f1, f2 etc of m times) of that dept
and purchased x items (n1 time x1, n2 times x2...) and returned y items (n1 time y1, n2 times y2...)
what is trip type look like ???..
'''

start_time = time.time()

# region data prep
def load_data(file, load_partial):
  if "test" in file:
    if load_partial:
      data = pd.read_csv(file, nrows=22757)
    else:
      data = pd.read_csv(file)
  else:  # train data
    if load_partial:
      data = pd.read_csv(file, nrows=1065201)
    else:
      data = pd.read_csv(file)

  print("loaded data of " + str(data.shape))

  return data




def add_features(data):
  '''
  Total number of items per visit
  Percentage of items purchased based on Department
  Percentage of items purchased based on Fineline
  Number Percentage of items purchased by UPC
  Count of different items purchased (based on UPC)
  Count of returned items Boolean for presence of returned item
  See more at: http://blog.nycdatascience.com/students-work/walmart-kaggle-trip-type-classification/#sthash.VbDVTNjs.dpuf

  Day of the week (expressed as an integer)
  Number of purchases per visit
  Number of returns per visit
  Number of times each department was represented in the visit
  Number of times each fineline number was represented in the visit -
  See more at: http://blog.nycdatascience.com/students-work/walmart-kaggle-trip-type-classification/#sthash.VbDVTNjs.dpuf

  '''

  TotalNumItems = data.groupby(['VisitNumber'], sort=False)['Upc'].count()
  TotalNumItems.name = 'ItemCount'

  # TotalItemsPurchased
  # TotalItemReturned
  return TotalNumItems


test_all_ids = []
test_non_empty_ids = []
test_empty_rows_ids = []

trip_types = [
  'TripType_3', 'TripType_4', 'TripType_5', 'TripType_6', 'TripType_7', 'TripType_8', 'TripType_9', 'TripType_12',
  'TripType_14', 'TripType_15', 'TripType_18', 'TripType_19', 'TripType_20', 'TripType_21', 'TripType_22',
  'TripType_23', 'TripType_24', 'TripType_25', 'TripType_26', 'TripType_27', 'TripType_28', 'TripType_29',
  'TripType_30',
  'TripType_31', 'TripType_32', 'TripType_33', 'TripType_34', 'TripType_35', 'TripType_36', 'TripType_37',
  'TripType_38',
  'TripType_39', 'TripType_40', 'TripType_41', 'TripType_42', 'TripType_43', 'TripType_44', 'TripType_999']


def uniqueCount(x):
  return np.unique(x)


scaler = preprocessing.StandardScaler()

def transform_add_features(data, file, vectorizer, vec, vecUpc, vecPickled, performPairWise, performChi):
  if "test" in file:
    global test_empty_rows_ids
    global test_non_empty_rows_ids
    test_non_empty_rows_ids = data['VisitNumber'][pd.notnull(data['DepartmentDescription'])]
    test_all_empty_rows_ids = data['VisitNumber'][data['DepartmentDescription'].isnull()]
    test_empty_rows_ids = list(set(test_all_empty_rows_ids) - set(test_non_empty_rows_ids))
    test_non_empty_rows_ids = list(set(test_non_empty_rows_ids))

  data_non_empty = data[pd.notnull(data['DepartmentDescription'])]
  print('dept. data shape', data_non_empty.shape)

  # Weekend vs Weekday
  weekdays = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
  X_non_empty = data_non_empty.replace({"Weekday": weekdays})

  X_non_empty['Day'] = (X_non_empty['Weekday'] > 4).astype(np.int32)

  day_type = X_non_empty.groupby(['VisitNumber'], sort=False)['Day'].first()
  day_type.name = 'DayType'
  print('day_type shape ', day_type.shape)

  #X_non_empty.drop(['Weekday'], axis=1)

  #total distinct items per visit
  total_items = X_non_empty.groupby(['VisitNumber'], sort=False)['Upc'].nunique()
  total_items.name = 'TotalUniqueItems'
  print('total_items (unique) shape ', total_items.shape)

  #total items per visit
  # total_all_items = X_non_empty.groupby(['VisitNumber'], sort=False)['ScanCount'].agg(lambda x: np.absolute(x).sum())
  # total_all_items.name = 'TotalAllItems'
  # print('total_all_items shape ', total_all_items.shape)

  #total # of unique dept. visited per visit
  dept_visit = X_non_empty.groupby(['VisitNumber'], sort=False)['DepartmentDescription'].nunique()
  dept_visit.name = 'TotalDeptVisit'
  print('dept_visit shape ', dept_visit.shape)

  #total # of unique fineline. visited per visit
  fineline_visit = X_non_empty.groupby(['VisitNumber'], sort=False)['FinelineNumber'].nunique()
  fineline_visit.name = 'TotalFinelineVisit'
  print('fineline_visit shape ', fineline_visit.shape)


  #total items purchased per visit
  total_purchase = X_non_empty.groupby(['VisitNumber'], sort=False)['ScanCount'].agg(lambda x: x[x > 0].sum())
  total_purchase.name = 'TotalPurchase'
  print('total_purchase shape ', total_purchase.shape)
  #
  # #total returned purchased per visit
  total_return = X_non_empty.groupby(['VisitNumber'], sort=False)['ScanCount'].agg(lambda x: np.absolute(x[x < 0]).sum())
  total_return.name = 'TotalReturn'
  print('total_return shape ', total_return.shape)

  # fineline0 = X_non_empty.groupby(['VisitNumber'], sort=False)['FinelineNumber'].
  # apply(lambda x: 1 if sum(x)==0 else 0)
  # fineline0.name = 'Fineline0'
  # print('fineline0 shape ', fineline0.shape)


  #total purchased items per visit
  #total_purchased = X_non_empty.groupby(['VisitNumber'], sort=False)['ScanCount'].count()
  #total_purchased.name = 'TotalPurchased'

  return_visit = X_non_empty.groupby(['VisitNumber'])['ScanCount'].sum()
  return_visit.name = 'ReturnVisit'
  return_visit.loc[return_visit <= 0] = -10
  return_visit.loc[return_visit > 0] = 10
  print('return_visit shape ', return_visit.shape)

  #TripType 3 mostly
  financial_type = ['FINANCIAL SERVICES']
  financial_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in financial_type]) / len(x))
  financial_visit.name = 'FinancialVisit'
  print('financial_visit shape ', financial_visit.shape)

  #pharmacy visit TripType 5 mostly
  med_type = ['PHARMACY RX', 'PHARMACY OTC']
  med_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in med_type]) / len(x))
  med_visit.name = 'MedlVisit'
  print('financial shape ', med_visit.shape)

  #TT 26
  HW_type = ['HARDWARE', 'PAINT AND ACCESSORIES']
  hw_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in HW_type]) / len(x))
  hw_visit.name = 'HWVisit'
  print('hw_visit shape ', hw_visit.shape)
  print(hw_visit.head(100))

  #TT 27
  garden_type = ['LAWN AND GARDEN', 'HORTICULTURE AND ACCESS']
  garden_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in garden_type]) / len(x))
  garden_visit.name = 'GardenVisit'
  print('garden_visit shape ', garden_visit.shape)
  #28
  sport_type = ['SPORTING GOODS']
  sport_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in sport_type]) / len(x))
  sport_visit.name = 'SportVisit'
  print('sport_visit shape ', sport_visit.shape)
  #31
  wireless_type = ['WIRELESS']
  wireless_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in wireless_type]) / len(x))
  wireless_visit.name = 'WirelessVisit'
  print('wireless_visit shape ', wireless_visit.shape)
  #32
  infant_type = ['INFANT CONSUMABLE HARDLINES', 'INFANT APPAREL']
  infant_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in infant_type]) / len(x))
  infant_visit.name = 'InfantVisit'
  print('infant_visit shape ', infant_visit.shape)

  #33
  household_type = ['HOUSEHOLD CHEMICALS/SUPP']  #HOUSEHOLD PAPER GOODS can be included as 0.5 weight
  household_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in household_type]) / len(x))
  household_visit.name = 'HouseHoldVisit'
  print('household_visit shape ', household_visit.shape)

  #34
  pets_type = ['PETS AND SUPPLIES']
  pets_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in pets_type]) / len(x))
  pets_visit.name = 'PetsVisit'
  print('pets_visit shape ', pets_visit.shape)

  #4
  pharma_otc_type = ['PHARMACY OTC']
  pharma_otc_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in pharma_otc_type]) / len(x))
  pharma_otc_visit.name = 'Pharma_otcVisit'
  print('pharma_otc_visit shape ', pharma_otc_visit.shape)

  #6
  wine_smoke_type = ['LIQUOR,WINE,BEER', 'CANDY, TOBACCO, COOKIES']
  wine_smoke_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in wine_smoke_type]) / len(x))
  wine_smoke_visit.name = 'WineSmokeVisit'
  print('wine_smoke_visit shape ', wine_smoke_visit.shape)

  #15
  celebration_type = ['CELEBRATION']
  celebration_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in celebration_type]) / len(x))
  celebration_visit.name = 'CelebrationVisit'
  print('celebration_visit shape ', celebration_visit.shape)

  toy_type = ['TOYS']
  toy_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in toy_type]) / len(x))
  toy_visit.name = 'ToyVisit'
  print('toy_visit shape ', toy_visit.shape)

  electronics_type = ['ELECTRONICS']
  electronics_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in electronics_type]) / len(x))
  electronics_visit.name = 'electronicsVisit'
  print('electronics_visit shape ', electronics_visit.shape)

  toy_type = ['TOYS']
  toy_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in toy_type]) / len(x))
  toy_visit.name = 'ToyVisit'
  print('toy_visit shape ', toy_visit.shape)

  automotive_type = ['AUTOMOTIVE']
  automotive_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
    agg(lambda x: sum([1.0 for i in x if i in automotive_type]) / len(x))
  automotive_visit.name = 'AutoVisit'
  print('automotive_visit shape ', automotive_visit.shape)

  frequent_depts = \
  ['SERVICE DELI',
  'GROCERY DRY GOODS',
  'PRODUCE',
  'FROZEN FOODS',
  'DSD GROCERY',
  'MEAT - FRESH & FROZEN',
  'COMM BREAD',
  'DAIRY',
  'BAKERY',
  'PRE PACKED DELI',
  'IMPULSE MERCHANDISE',
  'PERSONAL CARE',
  'HOUSEHOLD PAPER GOODS',
  'BEAUTY',
  'MENS WEAR',
  'LADIESWEAR',
  'CELEBRATION',
  'COOK AND DINE',
  'OFFICE SUPPLIES',
  'HOME MANAGEMENT'
  'SHOES',
  'FABRICS AND CRAFTS',
  'BATH AND SHOWER',
  'GIRLS WEAR, 4-6X  AND 7-14'
  'HOME DECOR',
  'BOYS WEAR'
  'JEWELRY AND SUNGLASSES',
  'SLEEPWEAR/FOUNDATIONS',
  'BEDDING',
  'BRAS & SHAPEWEAR',
  'MEDIA AND GAMING',
  'SEAFOOD',
  'ACCESSORIES',
  'BOOKS AND MAGAZINES',
  '1-HR PHOTO',
  'PLAYERS AND ELECTRONICS']

  dept_visits = []
  for dept in frequent_depts:
    dept_type = [dept]
    dept_visit = X_non_empty.groupby(['VisitNumber'])['DepartmentDescription']. \
      agg(lambda x: sum([1.0 for i in x if i in dept_type]) / len(x))
    dept_visit.name = dept+'Visit'
    dept_visits.append(dept_visit)
    print(dept_visit.name +' shape ', dept_visit.shape)

  #get all the  upc counts > n

  n = 500
  fine_count = X_non_empty[X_non_empty['DepartmentDescription']!='PHARMACY RX']['FinelineNumber'].value_counts()
  fine_list = fine_count.index[fine_count>n].values

  if 'train' in file:
    fineline_notin_test =\
     [3082.0, 524.0, 3599.0, 6674.0, 6680.0, 8731.0, 1566.0, 3616.0, 5668.0, 7716.0, 551.0, 4135.0, 7721.0, 45.0,
     7216.0, 2099.0, 5172.0, 7731.0, 567.0, 3133.0, 9792.0, 3648.0, 7235.0, 6213.0, 4685.0, 7246.0, 2128.0, 593.0, 6227.0,
     3673.0, 1627.0, 7773.0, 2656.0, 8801.0, 7781.0, 6247.0, 5739.0, 4207.0, 6256.0, 1145.0, 1658.0, 4742.0, 4247.0, 1691.0,
     4255.0, 8352.0, 6304.0, 3234.0, 7335.0, 2734.0, 6319.0, 7351.0, 3256.0, 3259.0, 1216.0, 6339.0, 8903.0, 6343.0, 2249.0,
     1226.0, 1737.0, 1739.0, 7372.0, 4814.0, 9937.0, 1235.0, 3795.0, 9942.0, 3798.0, 8408.0, 8185.0, 8417.0, 8930.0, 5859.0,
     1765.0, 8938.0, 9964.0, 238.0, 6382.0, 245.0, 3325.0, 6397.0, 6909.0, 6913.0, 5380.0, 3336.0, 3338.0, 4368.0, 2328.0,
     7453.0, 2847.0, 7460.0, 3365.0, 2342.0, 5415.0, 6453.0, 9527.0, 1336.0, 8505.0, 3385.0, 830.0, 5442.0, 2884.0, 1862.0,
     1864.0, 332.0, 1874.0, 1875.0, 6482.0, 1877.0, 8539.0, 3421.0, 8036.0, 8554.0, 4460.0, 7026.0, 7028.0, 373.0, 8565.0,
     7046.0, 7048.0, 8072.0, 6029.0, 4495.0, 8087.0, 8088.0, 7065.0, 8605.0, 3999.0, 7078.0, 4526.0, 4539.0, 5052.0, 3517.0,
     5564.0, 7112.0, 7625.0, 5066.0, 5578.0, 7117.0, 9175.0, 7130.0, 6623.0, 6629.0, 3046.0, 3047.0, 4597.0, 2039.0, 1529.0,
     3580.0]
    fine_list = list(set(fine_list) - set(fineline_notin_test))

  fl_visits = []
  for fl in fine_list:

    fl_type = [fl]
    fl_visit = X_non_empty.groupby(['VisitNumber'])['FinelineNumber']. \
      agg(lambda x: sum([1.0 for i in x if i in fl_type]) / len(x))
    fl_visit.name = str(fl)+'Visit'
    fl_visits.append(fl_visit)
    print(fl_visit.name +' shape ', fl_visit.shape)

  print('X_non_empty shape after replacing Weekday with 0/1', X_non_empty.shape)
  X = data_non_empty  #data.drop(['ScanCount', 'FinelineNumber'], axis=1)

  dept_desc = X['DepartmentDescription']
  dept_type = dict()

  #assign weekdays to numeric numbers


  num_features = ['VisitNumber']  #,'Weekday','Upc','ScanCount','FinelineNumber']
  if 'train' in file:
    num_features = ['TripType'] + num_features

  num_x = X[num_features].as_matrix()
  num_x = np.nan_to_num(num_x)

  print('num_features has NaN? ', num_x[np.isnan(num_x)])

  #cols_to_retain = ['DepartmentDescription']
  #cat_df = [ cols_to_retain ]
  #cat_dict = cat_df.T.to_dict().values()

  #cat_dict = X.drop(['TripType', 'VisitNumber', 'Weekday', 'Upc', 'ScanCount','FineLineNumber' ], axis = 1).T.to_dict().values()
  #print(X.columns)

  cat_data_dept_desc = X[['DepartmentDescription']]

  cat_data_dept_desc = cat_data_dept_desc.fillna('NA')
  #cat_data_dept_desc = cat_data_dept_desc[cat_data_dept_desc[]]
  ##print('vec_x_cat has NaN? ', cat_data_dept_desc[pd.isnull(cat_data_dept_desc).any(axis=1)])


  print('x converting to dict for vectorization...')
  cat_dict = cat_data_dept_desc.T.to_dict().values()

  #print(type(cat_dict))
  #print(cat_dict)
  vec_x_cat = None
  print('x vectorizing dept....')
  if 'train' in file:
    vec_x_cat = vectorizer.fit_transform(cat_dict)
    if vecPickled:
      with open('./trip_type_vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)
      f.close()
  elif 'test' in file:
    if vecPickled:
      with open('./trip_type_vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)
    vec_x_cat = vectorizer.transform(cat_dict)

  print('vec_x_cat shape before tf-idf', vec_x_cat.shape)
  print(type(vec_x_cat))

  # if 'train' in file:
  #   vec_x_cat = scaler.fit_transform(np.log(1+vec_x_cat))
  # if 'test' in file:
  #   vec_x_cat = scaler.transform(np.log(1+vec_x_cat))

  print('performing tf-idf...')
  tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
  vec_x_cat = tfidf.fit_transform(vec_x_cat)
  vec_x_cat = vec_x_cat.toarray()
  print(vec_x_cat)
  print('vec_x_cat shape after tf-idf', vec_x_cat.shape)
  print(type(vec_x_cat))
  print('vec_x_cat[3] tf-idf', vec_x_cat[3])
  print('vec_x_cat has NaN? ', np.isnan(vec_x_cat))

  #transformring sparse dept. counts
  #vec_x_cat = np.log(1+vec_x_cat)

  print('x vec feature names : ', vectorizer.get_feature_names())
  features_list = vectorizer.get_feature_names()
  features_list = num_features + features_list

  #note: getting memory error on a 8GB machine
  if performPairWise:
    print('calculating pairwise distances among dept. ...\n',
          pairwise_distances(vec_x_cat, metric='euclidean'))

  x_append = np.append(num_x, vec_x_cat, axis=1)
  print('x shape after appending', x_append.shape)
  print('x column names after appending', x_append.dtype.names)

  dfV = pd.DataFrame(data=x_append, columns=features_list)

  #feature selection and sparse -> dense
  #'DepartmentDescription=HEALTH AND BEAUTY AIDS' appears only twice 36 and 999 where Beauty is also there


  #dept scan fill

  didx = X.groupby(['DepartmentDescription'])['VisitNumber'].agg(lambda x: list(x.index))

  d_names = vectorizer.get_feature_names()

  x_rows = X.shape[0]
  #filling Dept count
  for c in d_names:
    c = c.replace('DepartmentDescription=','',1)
    dix = [i for i in didx[c] if i < x_rows]
    dfV.loc[dix, c] = X.loc[dix, 'ScanCount']
  dfV = dfV.fillna(0) #incase scan count was NaN
  dfV.loc[dfV['DepartmentDescription=HEALTH AND BEAUTY AIDS'] > 0, 'DepartmentDescription=BEAUTY'] \
    = dfV['DepartmentDescription=HEALTH AND BEAUTY AIDS'] + dfV['DepartmentDescription=BEAUTY']

  dfV = dfV.drop(['DepartmentDescription=HEALTH AND BEAUTY AIDS'], axis=1)


  #create FinelineNumber feature
  #only the non RX ones
  fileline_count = X[X['DepartmentDescription']!='PHARMACY RX']['FinelineNumber'].value_counts()
  indx = fileline_count[fileline_count > 1000].index

  print(indx.values.astype(np.int64))
  int_filelineNum = indx.values.astype(np.int64)  #was int
  global finelineUniqueDict
  finelineUniqueDict = [{str(x): 0} for x in int_filelineNum]
  #print(finelineUniqueDict)

  fineArray = None
  if 'train' in file:
    fineArray = vec.fit_transform(finelineUniqueDict)
  if 'test' in file:
    fineArray = vec.transform(finelineUniqueDict)

  print('fineline features',vec.get_feature_names(finelineUniqueDict))
  fineArray = np.zeros((X.shape[0], fineArray.shape[1]))

  f_names = vec.get_feature_names()
  flDf = pd.DataFrame(fineArray, columns=f_names)

  print('flDf shape', flDf.shape)
  #fill NaNs with 12345 ids
  X['FinelineNumber'] = X['FinelineNumber'].fillna(12345).astype(np.int64)
  fidx = X.groupby(['FinelineNumber'])['VisitNumber'].agg(lambda x: list(x.index))

  fnames_int = list(map(int, f_names))
  x_rows = X.shape[0]
  #filling FinelineNumber count
  for c in fnames_int:
    fix = [i for i in fidx[c] if i < x_rows]
    flDf.loc[fix, str(c)] = X.loc[fix,'ScanCount']

  flDf = flDf.fillna(0) #incase scan count was NaN
  ################################################


  #create UPC feature count >100 has about 560
  #get the non RX type
  upc_count = X[X['DepartmentDescription']!='PHARMACY RX']['Upc'].value_counts()
  indx = upc_count[upc_count > 500].index

  print(indx.values.astype(np.int64))
  int_UpcNum = indx.values.astype(np.int64)  #was int
  global upcUniqueDict
  upcUniqueDict = [{str(x): 0} for x in int_UpcNum]
  #print(finelineUniqueDict)

  upcArray = None
  if 'train' in file:
    upcArray = vecUpc.fit_transform(upcUniqueDict)
  if 'test' in file:
    upcArray = vecUpc.transform(upcUniqueDict)

  upcArray = np.zeros((X.shape[0], upcArray.shape[1]))

  f_names = vecUpc.get_feature_names()
  uDf = pd.DataFrame(upcArray, columns=f_names)

  print('uDf shape', uDf.shape)
  #fill NaNs with 12345 ids
  X['Upc'] = X['Upc'].fillna(123).astype(np.int64)
  fidx = X.groupby(['Upc'])['VisitNumber'].agg(lambda x: list(x.index))

  #X['Upc'] = X['Upc'].fillna(0).astype(np.uint64)
  #fidx = X.groupby(['Upc'])['VisitNumber'].agg(lambda x : list(x.index))


  unames_int = list((int, f_names))
  x_rows = X.shape[0]
  #filling FinelineNumber count
  for c in unames_int:
    fix = [i for i in fidx[c] if i < x_rows]
    uDf.loc[fix, str(c)] = 1
  uDf = uDf.fillna(0)


  #concat fileNum with Dept

  dfV = pd.concat([dfV, flDf, uDf], axis=1, join='inner')	


  print('dfV shape', dfV.shape)
  #print('dfV columns', dfV.columns)
  #print(dfV.head(100))
  #dfV.to_csv('DeptFinelineFeature.csv')
  #print('null rows',dfV[pd.isnull(dfV)])

  labels = None
  if 'train' in file:
    trip_type = dfV.groupby(['VisitNumber'], sort=False)['TripType'].first()
    dfV = dfV.drop(['TripType'], axis=1)
    labels = trip_type
    print('# of labels', trip_type.shape)

  print('dfV without TripType shape', dfV.shape)

  #print('dfV visit - dept_ACCESSORIES \n',dfV[['VisitNumber','DepartmentDescription=SHOES', 'DepartmentDescription=PAINT AND ACCESSORIES',
  #                                             'DepartmentDescription=ACCESSORIES']].head(4))

  dfVUnique = dfV.groupby(['VisitNumber']).sum()
  print('# of unique visit - dept-vec before adding new features ', dfVUnique.shape)

  #add other features
  non_dep_features = [day_type, total_items, dept_visit, fineline_visit, total_purchase, total_return, return_visit]
  dept_trip_features = \
  [financial_visit, med_visit,hw_visit, garden_visit, sport_visit, wireless_visit,
   infant_visit, household_visit, pets_visit, pharma_otc_visit,wine_smoke_visit,
   celebration_visit, toy_visit, automotive_visit]

  all_features = non_dep_features + dept_trip_features + dept_visits + [dfVUnique] 
  df_all_features = pd.concat(all_features, axis=1, join='inner')

  print('# of unique visit - dept-vec', df_all_features.shape)
  print('data types of df_all_features', df_all_features.dtypes)
  #df_all_features.head(5000).to_csv('df_all_features.csv')
  global features
  features = df_all_features.columns

  if 'test' in file:
    return df_all_features

  if 'train' in file and performChi:
    print('performing chi test... \n', chi2(df_all_features.iloc[:, 20:], labels))

  return df_all_features, labels


def standardize_data(X):
  mean = X.mean(axis=0)
  X -= mean
  std = X.std(axis=0)
  X /= std
  standardized_data = X
  return standardized_data

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

def impute_data(non_empty_data):
  imp.fit(non_empty_data)
  X = imp.transform(non_empty_data)  # 23.2592765571 (better)
  return X  # non_empty_data.fillna(0) #23.2628586644 # X

vectorizer = DV(sparse=False)
vec = DV(sparse=False)
vecUpc = DV(sparse=False)

def prepare_train_data(file_path, load_Partial, analyisPerformed, vecPickled, performPairWise, performChi):
  print("preparing training data...")
  train_data = load_data(file_path, load_Partial)

  if analyisPerformed:
    analyze_plot_data(train_data, 'train')

  null_rows = train_data[pd.isnull(train_data).any(axis=1)]
  train_data.fillna(0)

  print('null rows_shapes', null_rows.shape)

  print("transforming data...")
  tx_data, labels = transform_add_features(train_data, file_path, vectorizer, vec, vecUpc, vecPickled, performPairWise, performChi)
  X_train = tx_data

  return X_train, labels


def prepare_test_data(file_path, load_partial, vecPickled, performPairWise, performChi):
  print("preparing test data...")
  test_data = load_data(file_path, load_partial)
  X = transform_add_features(test_data, file_path, vectorizer, vec, vecUpc, vecPickled, performPairWise, performChi)
  test_data.fillna(0)

  return X  # test_input

# endregion data prep

# region train

def cv_score(clf, X, y):
  print("cross validating model...")
  scores = cross_validation.cross_val_score(clf, X, y.values, cv=3, scoring='log_loss')
  return abs(sum(scores) / len(scores))


def scorer(estimator, X, y):
  # todo: fill and use in CV with https://www.kaggle.com/c/walmart-recruiting-trip-type-classification/details/evaluation
  pass


def cross_validate_model(model, X_train, y_train):
  cvs = cv_score(model, X_train, y_train)
  print("Log loss on cross validation set: " + str(cvs))


model = None


def pickle_model(model, model_index):
  # pickle model
  with open('./pickled_model/' + str(model_index) + '_trip_type.pickle', 'wb') as f:
    pickle.dump(model, f)
  f.close()


def unpickle_model(file):
  with open(file, 'rb') as f:
    model = pickle.load(f)
  return model


def split_train_data(train_input, labels, t_size):
  # train_test does shuffle and random splits
  if len(train_input) != len(labels):
    print('X #{0} and y#{1} does not match'.format(len(train_input), len(labels)))
  else:
    print('X #{0} and y#{1} does match'.format(train_input.shape[0], len(labels)))

  X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    train_input, labels, test_size=t_size, random_state=1)
  return X_train, X_test, y_train, y_test


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
  """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
  predictions = np.clip(y_pred, eps, 1 - eps)

  # normalize row sums to 1
  predictions /= predictions.sum(axis=1)[:, np.newaxis]

  actual = np.zeros(y_pred.shape)
  n_samples = actual.shape[0]
  actual[np.arange(n_samples), y_true.astype(int)] = 1
  vectsum = np.sum(actual * np.log(predictions))
  loss = -1.0 / n_samples * vectsum
  return loss

le = preprocessing.LabelEncoder()
def test_model(model, model_type, X_test, y_test, testBiasVarience):
  # print('y_test ', y_test)
  #print('y_pred ', y_pred)

  #print("Multi-class Log loss on holdout test set", multiclass_log_loss(y_test.values, y_pred))
  if testBiasVarience:
    i = 1
    for mdl in model:
      y_pred = mdl.predict_proba(X_test)
      print(y_pred[0])
      print("Log loss {0} on holdout test set for model{1}".format(log_loss(y_test.values, y_pred), i))
      i += 1
  else:
    if model_type == 'xGB':
        print('xgb model testing...')
        x_test = xgb.DMatrix(X_test.values)
        y_pred = model.predict(x_test)
    else:
      y_pred = model.predict_proba(X_test)
    print("Log loss on holdout test set", log_loss(y_test.values, y_pred))

def feature_selection(model, features):
  # feature engineering
  print("feature selection...")
  # print(features)
  print("feature importance", model.feature_importances_)
  min_index = np.argmin(model.feature_importances_)
  print("min feature : ({0}){1}, score {2} ".format(min_index, features[min_index], min(model.feature_importances_)))

def parameter_search(model, X, y):
  param_grid = {
    'learning_rate': [0.07],  # 0.07 best so far
    'max_depth': [10],  #10 best so far
    #'n_estimators' : [60, 100, 150],
    'subsample': [0.5],  #0.5 best so far
    'max_features': [0.3],  #0.3 best so far
    #'min_samples_split': [4, 8, 12],
    #'min_samples_leaf': [3, 5, 8]
  }

  # gs_cv = GridSearchCV(model,param_grid,n_jobs=-1, scoring='log_loss').fit(X,y)

  rs_cv = RandomizedSearchCV(model, param_distributions=param_grid, n_jobs=-1, n_iter=20,
                             verbose=5, scoring='log_loss').fit(X, y)

  pram_cv = rs_cv
  print('grid scores , \n', rs_cv.grid_scores_)
  print("best parameters {0} from grid search gave score {1} ".format(pram_cv.best_params_, pram_cv.best_score_))
  print('best estimator , \n', rs_cv.best_estimator_)


def train_model(seed, X_train, y_train, X_train_test, y_train_test, n_est, max_depth, cvPerformed, model_type,
                performGridSearch, testBiasVarience):
  # RF 100 = 0.99 400=0.98 -- 800=0.97
  rf_c = ensemble.RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, min_samples_leaf=3, bootstrap=False,
                                         random_state=seed, max_features='sqrt', criterion='gini', n_jobs=-1)

  # ET 400 gives 1.06
  et_c = ensemble.ExtraTreesClassifier(n_estimators=n_est, max_depth=max_depth, min_samples_leaf=3, random_state=seed,
                                       max_features='sqrt', criterion='gini', n_jobs=-1)

  # gives 0.89 on 150 estimators
  gb_c = ensemble.GradientBoostingClassifier(n_estimators=n_est, learning_rate=0.07, max_depth=7, verbose=1,  #5 ideal
                                             min_samples_leaf=3, random_state=seed, max_features=0.3, subsample=0.5)
  #max feature was 0.2, try subsample=0.5

  #neighbors = 200 gives 1.34 &   400 gives 1.307
  knn = KNeighborsClassifier(n_neighbors=n_est, weights='uniform')  #try weight with 'distance'

  #LR LB score = 2.77936
  #clf = linear_model.LogisticRegression(max_iter=n_est, random_state=seed)

  #AdaBoost LB score =  #didn't use ensemble , just one
  ab_c = ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=n_est, algorithm='SAMME',
                                     learning_rate=0.1, random_state=seed)

  if model_type == 'RF':
    clf = rf_c
  if model_type == 'ET':
    clf = et_c
  if model_type == 'GB':
    clf = gb_c
  if model_type == 'AB':
    clf = ab_c
  if model_type == 'KNN':
    clf = knn

  #csr = sp.sparse.csr_matrix(X_train.values)
  #dtrain = xgb.rix(csr)

  if model_type == 'xGB':
    y_train = le.fit_transform(y_train.values)
    y_train_test = le.transform(y_train_test.values)
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    #X_train, y_train, test_size=0.09, random_state=1)

    lbls = y_train
    lbls_holdout = y_train_test

    dtrain = xgb.DMatrix(X_train.values, label=lbls)
    dxholdout = xgb.DMatrix(X_train_test.values, label=lbls_holdout)

    num_round = n_est
    #try tuning parameters subsample =0.5 , max_depth - try 10? and gamma, try eta 0.01 with 4000 rounds
    #and colsample_bytree, min_child_weight
    param = {'bst:max_depth':8, 'bst:eta':0.01, 'silent':1, 'objective':'multi:softprob', 'num_class':38 ,
             'max_delta_step':1, 'gamma':2, 'subsample':0.75, 'min_child_weight':3, 'colsample_bytree':0.5}
    param['eval_metric'] = 'mlogloss'
    evallist  = [(dtrain,'train'),(dxholdout,'holdout')]
    #
    bst = xgb.train( param, dtrain, num_round, evallist, early_stopping_rounds=50)
    #label = y_train

    ypred = bst.predict(dtrain)
    print("training error {0} xgboost...".format(log_loss(y_train, ypred)))

    print('best iteration:',bst.best_iteration)
    print('best score: ',bst.best_score)
    print('best ntree_limit ', bst.best_ntree_limit)

    return bst

  if cvPerformed:
    cross_validate_model(clf, X_train, y_train)

  if performGridSearch:
    parameter_search(clf, X_train, y_train)

  if testBiasVarience:
    bv_models = []
    for n_e in range(50, 250, 50):
      #prams = {'n_estimators':n_e}
      clf = clf.set_params(n_estimators=n_e)
      print("training model_{0}...".format(seed))
      model = clf.fit(X_train, y_train)
      bv_models.append(clf)
      print("training error {0} for model{1}...".format(log_loss(y_train.values, clf.predict_proba(X_train)), int(n_e / 50)))
      #print('trainging oob score ', model.oob_score_)
      #print('training OOB decision func.. ', model.oob_decision_function_)
      if model_type == 'GB':
        print('training OOB improvement ', model.oob_improvement_)
  else:
    print("training model_{0}...".format(seed))
    model = clf.fit(X_train, y_train)
    print("training error {0} for model{1}...".format(log_loss(y_train.values, clf.predict_proba(X_train)), seed))
    #print('trainging OOB score ', model.oob_score_)
    if model_type == 'GB':
      print('training OOB improvement ', model.oob_improvement_)

  if testBiasVarience:
    return bv_models



  return model


# endregion train

def analyze_plot_data(data, type):
  print('summary of data', data.describe())

  test_file_path = "./test/test.csv"
  dtest = load_data(test_file_path, False)

  trainUpc = data['Upc'].unique().tolist()
  testUpc = dtest['Upc'].unique().tolist()
  print(trainUpc) ; print(testUpc)

  tdiff = set(trainUpc)-set(testUpc)
  print('Upc in train not in test', tdiff)

  trainUpc = data['FinelineNumber'].unique().tolist()
  testUpc = dtest['FinelineNumber'].unique().tolist()
  print(trainUpc) ; print(testUpc)

  tdiff = set(trainUpc)-set(testUpc)

  text_file = open("FinelineNotInTest.txt", "w")
  text_file.write(str(tdiff))
  text_file.close()

  #print(type(trainUpc)) ; print(type(testUpc))
  print('Fineline in train not in test', tdiff)


  #data['DepartmentDescription'].value_counts().to_csv('DeptCount.csv')
  # ttset = set(data['TripType'].tolist())
  # for tt in ttset:
  # ttdept = data[data['TripType']== tt]['DepartmentDescription'].value_counts()
  #   print('tt_dept frequencies \n', ttdept)
  #   ttdept.to_csv('tt'+str(tt)+'_dept_freq.csv')

  #finelineDept = data.groupby(['FinelineNumber'])['DepartmentDescription'].apply(lambda x: (len(set(x)), list(set(x))))
  #finelineDept.to_csv('DeptCount.csv')
  #print('finelineDept \n');
  #print(finelineDept)

  #upcDept = data.groupby(['Upc'])['DepartmentDescription'].apply(lambda x: (len(set(x)), list(set(x))))
  #upcDept.to_csv('Upc_DeptMap.csv')
  #print('upcDeptMap \n');print(upcDept)

  '''
  GROCERY DRY GOODS
  DSD GROCERY
  PRODUCE
  DAIRY
  PERSONAL CARE
  IMPULSE MERCHANDISE
  HOUSEHOLD CHEMICALS/SUPP
  PHARMACY OTC
  FROZEN FOODS
  HOUSEHOLD PAPER GOODS
  COMM BREAD
  BEAUTY
  MENS WEAR
  FINANCIAL SERVICES
  INFANT CONSUMABLE HARDLINES
  SERVICE DELI

  '''
  grocery_type = ['GROCERY DRY GOODS', 'DSD GROCERY', 'DAIRY', 'PRODUCE', 'HOUSEHOLD CHEMICALS/SUPP',
                  'HOUSEHOLD PAPER GOODS',
                  'PERSONAL CARE', 'FROZEN FOODS', 'COMM BREAD', 'PHARMACY OTC', 'PRE PACKED DELI',
                  'IMPULSE MERCHANDISE',
                  'MEAT - FRESH & FROZEN', 'PETS AND SUPPLIES', 'BEAUTY', 'CANDY, TOBACCO, COOKIES',
                  'INFANT CONSUMABLE HARDLINES',
                  'SERVICE DELI', 'BAKERY', 'COOK AND DINE', 'LIQUOR,WINE,BEER', 'CELEBRATION', 'SEAFOOD']


  #FineDeptFreq = data[data['DepartmentDescription'].isin(grocery_type)].groupby(['DepartmentDescription'])\
  #  ['FinelineNumber'].apply(lambda x: (len(set(x)), list(set(x))))

  FineDeptFreq = data.groupby(['DepartmentDescription'])\
    ['FinelineNumber'].apply(lambda x: len(set(x)))

  FineDeptFreq.to_csv('DeptFineCount.csv')

  #  grocery = data.groupby(['VisitNumber', 'TripType'])['DepartmentDescription'].\
  #   agg(lambda x: (#x.value_counts().idxmax() if x.value_counts()>0 else 0,
  #                  #x.value_counts().max() if x.value_counts()>0 else 0,
  #                  sum([1 for i in x if i in grocery_type])/len(x)))

  #print('grocery \n', grocery)
  #grocery.to_csv('grocery.csv')

  financial_type = ['FINANCIAL SERVICES']

  # financial = data[data['TripType']==3].groupby(['VisitNumber', 'TripType'])['DepartmentDescription'].\
  #   agg(lambda x: (#x.value_counts().idxmax() if x.value_counts()>0 else 0,
  #                  #x.value_counts().max() if x.value_counts()>0 else 0,
  #                  sum([1 for i in x if i in financial_type])/len(x)))
  #
  #
  # print('financial \n', financial)

  #data['TripType'].value_counts().to_csv('trip_type_count.csv')

  #triptypeScan = data.groupby(['VisitNumber', 'TripType'])[['ScanCount']].sum()
  triptypeScan = data.groupby(['VisitNumber'])['ScanCount'].sum()
  print(triptypeScan)

  #returnTrip = triptypeScan[triptypeScan['ScanCount'] <= 0]
  #print(returnTrip)

  #triptypeScan.loc[triptypeScan['ScanCount'] <= 0] = -10
  #print(triptypeScan[triptypeScan['ScanCount'] <= 1])

  #returnTrip.to_csv('returnTrip.csv')

  triptypeDept = data.groupby(['TripType'])['DepartmentDescription'].apply(lambda x: (len(set(x)), list(set(x))))
  triptypeDept.to_csv('triptypeDeptMap.csv')
  print('triptype-Dept \n');
  print(triptypeDept)

  dept_triptype = data.groupby(['DepartmentDescription'])['TripType'].apply(lambda x: (len(set(x)), list(set(x))))
  dept_triptype.to_csv('dept_triptype_Map.csv')

  print('number of distinct UPC/items: ', len(set(data['Upc'])))
  print('frequencies of distinct UPC/items: \n', data['Upc'].value_counts())

  print('number of distinct DepartmentDescription: ', len(set(data['DepartmentDescription'])))
  print('frequencies of distinct DepartmentDescription: \n', data['DepartmentDescription'].value_counts())

  print('number of distinct FinelineNumber: ', len(set(data['FinelineNumber'])))

  data_fine0_tripNot999 = data[(data['FinelineNumber'] == 0) & (data['TripType'] != 999)]
  print('number of distinct FinelineNumber with dept: \n');
  print(data_fine0_tripNot999[data_fine0_tripNot999['DepartmentDescription'] != 'FINANCIAL SERVICES'])

  #check if all Fineline# in a visit is 0, then triptype 999
  #allfine0 = data.groupby(['VisitNumber'])['FinelineNumber'].apply(lambda x: 1 if sum(x)==0 else 0 )
  #print(allfine0[allfine0==1])

  #get only the non RX ones
  fileline_count = data['FinelineNumber'].value_counts()
  indx = fileline_count[fileline_count > 2000].index

  print(indx.values.astype(int))
  int_filelineNum = indx.values.astype(int)
  finelineUniqueDict = [{str(x): 0} for x in int_filelineNum]
  print(finelineUniqueDict)
  #print(pd.get_dummies(int_filelineNum))
  vec = DV(sparse=False)
  fineArray = vec.fit_transform(finelineUniqueDict)

  fineArray = np.zeros((data.shape[0], fineArray.shape[1]))
  feature_names = vec.get_feature_names()
  print(feature_names)
  flDf = pd.DataFrame(fineArray, columns=feature_names)
  flDf = pd.concat([data['VisitNumber'], flDf], axis=1, join='inner')
  print(flDf.shape)
  print(flDf.head(5))


  #flWithAllDf = pd.concat([data[['VisitNumber', 'FinelineNumber']], flDf], join='inner', axis=1)

  #print(flWithAllDf.head(20))
  #print(flWithAllDf.columns)
  fill_time = time.time()
  f_names = vec.get_feature_names()

  data['FinelineNumber'] = data['FinelineNumber'].fillna(-1).astype(np.int64)
  fidx = data.groupby(['FinelineNumber'])['VisitNumber'].agg(lambda x: list(x.index))

  fnames_int = list(map(int, f_names))
  x_rows = data.shape[0]
  #filling FinelineNumber count
  for c in fnames_int:
    fix = [i for i in fidx[c] if i < x_rows]
    flDf.loc[fix, str(c)] = 1
  print('fill time ', time.time() - fill_time)

  flDfUnique = flDf.groupby(['VisitNumber']).sum()

  print(vec.get_feature_names())
  print(fineArray)

  #analyzing UPC

  upc_count = data['Upc'].value_counts()
  upc_count.to_csv('upc_count.csv')
  indx = upc_count[upc_count > 500].index

  print(indx.values.astype(np.int64))
  int_UpcNum = indx.values.astype(np.int64)
  upcUniqueDict = [{str(x): 0} for x in int_UpcNum]
  print(upcUniqueDict)
  #print(pd.get_dummies(int_filelineNum))
  vecU = DV(sparse=False)
  uArray = vecU.fit_transform(upcUniqueDict)

  uArray = np.zeros((data.shape[0], uArray.shape[1]))
  feature_names = vecU.get_feature_names()
  print(feature_names)
  uDf = pd.DataFrame(uArray, columns=feature_names)
  uDf = pd.concat([data['VisitNumber'], uDf], axis=1, join='inner')
  print(uDf.shape)
  print(uDf.head(5))


  #flWithAllDf = pd.concat([data[['VisitNumber', 'FinelineNumber']], flDf], join='inner', axis=1)

  fill_time = time.time()
  f_names = vecU.get_feature_names()

  data['Upc'] = data['Upc'].fillna(-1).astype(np.int64)
  fidx = data.groupby(['Upc'])['VisitNumber'].agg(lambda x: list(x.index))

  fnames_int = list(map(int, f_names))
  x_rows = data.shape[0]
  #filling FinelineNumber count
  for c in fnames_int:
    fix = [i for i in fidx[c] if abs(i) < x_rows]
    uDf.loc[fix, str(c)] = 1
  print('fill time ', time.time() - fill_time)

  #flWithAllDf.drop(['FinelineNumber'], axis=1)
  uDfUnique = uDf.groupby(['VisitNumber']).sum()

  print(flDfUnique.head(100))
  uDfUnique.to_csv('UpcSum.csv')
  print(vecU.get_feature_names())
  print(uArray)

  print('frequencies of distinct FinelineNumber: \n');
  print(data['FinelineNumber'].value_counts())

  dept_fineline = data.groupby(['DepartmentDescription'])['FinelineNumber'].apply(lambda x: (len(set(x)), list(set(x))))
  print('dept_fineline \n', dept_fineline)

  data = data.drop(['Weekday', 'DepartmentDescription'], axis=1)

  # upc = data['Upc'].tolist()
  # upc_count = [(x, upc.count(x)) for x in set(upc)]
  # for upc in upc_count:
  # print('Upc items '.format(upc[0], upc[1]))

  # if data series data
  if isinstance(data, pd.Series):
    data.hist(color='green', alpha=0.5, bins=50, orientation='horizontal', figsize=(16, 8))
    # plt.title("distribution of samples in -> " + data.name)
    #plt.ylabel("frequency")
    #plt.xlabel("value")
    pl.suptitle("trip_type_" + type + "_" + data.name)
    #plt.show()
    file_to_save = "trip_type_" + type + "_" + data.name + ".png"
    path = os.path.join("./charts/", file_to_save)
    plt.savefig(path)
  else:  # plot all data features/columns
    for i in range(0, len(data.columns)):
      #plt.title("distribution of samples in -> " + data.columns[i])
      #plt.ylabel("frequency")
      #plt.xlabel("value")
      data[data.columns[i]].hist(color='green', alpha=0.5, bins=50, figsize=(16, 8))
      pl.suptitle("trip_type" + type + "_" + data.columns[i])
      plt.show()
      file_to_save = "trip_type" + type + "_" + data.columns[i] + ".png"
      path = os.path.join("./charts/", file_to_save)
      plt.savefig(path)


      #plt.figure()
      #print(data.min())
      #basic statistics of the data
      #print(data.describe())

      #data.hist(color='k', alpha=0.5, bins=25)
      #plt.hist(data, bins=25, histtype='bar')
      #plt.title(data.columns[0]+"distribution in train sample")
      #plt.savefig(feature_name+".png")
      #plt.show()  #test


def write_prediction(test_y, model_index):
  print("writing to file{0}....".format(model_index))
  predictionDf = pd.DataFrame(index=test_non_empty_rows_ids, columns=trip_types, data=test_y)
  # predict 0 for empty rows/ids
  print('total non empty samples predicted', predictionDf.shape)
  empty_test_y = np.asanyarray([[0.0 for _ in range(38)] for _ in test_empty_rows_ids])
  # assuming all unique dept = NULL rows are Category Other/999
  empty_test_y[:, 37] = 0.75
  print('empty_test_y[0] ', empty_test_y[0])
  emptyRowsDf = pd.DataFrame(index=test_empty_rows_ids, columns=trip_types, data=empty_test_y)

  totalDf = pd.concat([predictionDf, emptyRowsDf])
  print('total samples predicted', totalDf.shape)

  totalUniqueDf = totalDf.groupby(level=0).max()
  totalDf.sort_index(inplace=True)
  print('total unique samples predicted', totalUniqueDf.shape)

  # print(type(test_ids))
  #print(type(test_ids[0]))
  #print(predictionDf.columns)
  # write file
  prediction_file = './' + str(model_index) + '_trip_type_prediction.csv'
  totalUniqueDf.to_csv(prediction_file, index_label='VisitNumber', float_format='%.6f')
  print("writing prediction to file Done")


def predict_test(model, model_type, model_index, test_input, isPickled):
  print("predicting using model_{0}....".format(model_index))

  # model = joblib.load("./pickled_model/rain2.pkl") if isPickled else model
  model = unpickle_model('./pickled_model/' + str(model_index) + '_trip_type.pickle') if isPickled else model
  test_y = None
  if model:
    if model_type =='xGB':
      xgb_test = xgb.DMatrix(test_input.values)
      test_y = model.predict(xgb_test)
    else:
      test_y = model.predict_proba(test_input)
    print('prediction done. test_y shape ', test_y.shape)
  else:
    print("no model found..")
  return test_y

def begin_train(n_parallel_models, isPickled, loadPartial, model_type, n_est, max_depth, cvPerformed,
                perform_model_test, t_split_size, analyisPerformed, feature_selection_ON, vecPickled,
                performGridSearch, performBlend, performPairWise, performChi, testBiasVarience):
  train_file_path = "./train/train.csv"
  train_input, labels = prepare_train_data(train_file_path, loadPartial, analyisPerformed, vecPickled,
                                           performPairWise, performChi)
  X_train, y_train = train_input, labels
  X_train_test, y_train_test = None, None

  print('X_train len{0} , y_train len{1}'.format(X_train.shape[0], y_train.shape[0]))

  if perform_model_test:
    X_train, X_train_test, y_train, y_train_test = split_train_data(train_input, labels, t_split_size)
    print(
      'X_train {0}, y_train {1}, X_train_test {2}, y_train_test {3} [WithSplit]'.format(X_train.shape, y_train.shape,
                                                                                        X_train_test.shape,
                                                                                        y_train_test.shape))
  else:
    print('X_train {0}, y_train{1} shape [NoSplit]'.format(X_train.shape, y_train.shape))

  rf_c, et_c, knn_c = None, None, None
  X_train_rest, y_train_rest = None, None
  if performBlend:
    rows = np.int32(X_train.shape[0] * 0.2)
    print('total row/samples to use for blending ', rows)
    X_blend = X_train.iloc[:rows, :]
    y_blend = y_train.iloc[:rows]
    print('training RF for blending...')
    rf_c = ensemble.RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=3, bootstrap=False,
                                           random_state=1, max_features='sqrt', criterion='gini', n_jobs=-1).fit(
      X_blend, y_blend)
    print('training log loss RF blending... ', log_loss(y_blend.values, rf_c.predict_proba(X_blend)))
    print('training ET for blending...')
    et_c = ensemble.ExtraTreesClassifier(n_estimators=400, max_depth=None, min_samples_leaf=3,
                                         random_state=2, max_features='sqrt', criterion='gini', n_jobs=-1).fit(X_blend,
                                                                                                               y_blend)
    # gb_c = ensemble.GradientBoostingClassifier(n_estimators=30, learning_rate=0.09, max_depth=10,verbose=1,
    #       min_samples_leaf=3, random_state=3, max_features='sqrt', subsample=0.5).fit(X_blend, y_blend)
    print('training log loss ET blending... ', log_loss(y_blend.values, et_c.predict_proba(X_blend)))

    print('training KNN for blending...')
    knn_c = KNeighborsClassifier(n_neighbors=150, weights='uniform').fit(X_blend, y_blend)  #use 200-400
    print('training log loss KNN blending... ', log_loss(y_blend.values, knn_c.predict_proba(X_blend)))

    print('training xgb for blending...')

    lbls = le.fit_transform(y_blend.values)
    xgbtrain = xgb.DMatrix(X_blend.values, label=lbls)
    num_round = 400
    param = {'bst:max_depth':8, 'bst:eta':0.1, 'silent':1, 'objective':'multi:softprob', 'num_class':38 ,
             'max_delta_step':1, 'gamma':2, 'subsample':0.9, 'min_child_weight':3, 'colsample_bytree':0.5}
    param['eval_metric'] = 'mlogloss'
    evallist  = [(xgbtrain,'train')]
    bst_c = xgb.train( param, xgbtrain, num_round, evallist, early_stopping_rounds=10)

    clf_list = [rf_c, et_c, knn_c, bst_c]

    X_train_rest = X_train.iloc[rows:, :]
    y_train_rest = y_train.iloc[rows:]

    lbls_rest = le.transform(y_train_rest.values)
    xgb_rest = xgb.DMatrix(X_train_rest.values, label=lbls_rest)

    X_train = np.append(X_train_rest, np.power(
      rf_c.predict_proba(X_train_rest) * et_c.predict_proba(X_train_rest) * knn_c.predict_proba(X_train_rest) *
      bst_c.predict(xgb_rest), (1 / 4.0)), axis=1)
    y_train = y_train_rest

  print('X_train shape{0} , y_train shape{1}'.format(X_train.shape, y_train.shape))
  models = Parallel(n_jobs=n_parallel_models, backend="threading")(
    delayed(train_model)(seed, X_train, y_train, X_train_test, y_train_test, n_est, max_depth, cvPerformed,
                         model_type, performGridSearch, testBiasVarience) for seed in range(1, n_parallel_models + 1))

  for model in models:
    print(type(model));
    if isPickled:
      print("pickling model...")
      pickle_model(model, models.index(model))
    if feature_selection_ON and model_type != 'KNN':
      feature_selection(model, features)
    if perform_model_test:
      if performBlend:
        X_train_test = np.append(X_train_test, np.power(
          rf_c.predict_proba(X_train_test) * et_c.predict_proba(X_train_test) * knn_c.predict_proba(X_train_test),
          (1 / 3.0)), 1)
      test_model(model, model_type, X_train_test, y_train_test, testBiasVarience)

  print('total training execution time(min) required: ', (time.time() - start_time) / 60)
  return models


def begin_test(n_parallel, models, model_type, isPickled, loadPartial, vecPickled, performPairWise, performChi):
  test_file_path = "./test/test.csv"
  X_test = prepare_test_data(test_file_path, loadPartial, vecPickled, performPairWise, performChi)

  if models is None:
    models = [None for _ in range(n_parallel)]
  indices = [i for i in range(n_parallel)]
  test_ys = Parallel(n_jobs=n_parallel, backend="threading")(delayed(predict_test)
                                                             (model, model_type, m_index, X_test, isPickled) for m_index, model in
                                                             zip(indices, models))

  Parallel(n_jobs=n_parallel, backend="threading")(delayed(write_prediction)
                                                   (test_y, m_index) for m_index, test_y in zip(indices, test_ys))


def ensemble_predictions(num_pred_files):
  print('ensembling predictions...')
  pdf1 = pd.DataFrame()
  pdf2 = pd.DataFrame()
  pdf3 = pd.DataFrame()
  pdf4 = pd.DataFrame()
  pdf5 = pd.DataFrame()
  pdf6 = pd.DataFrame()
  pdf7 = pd.DataFrame()
  pdf8 = pd.DataFrame()
  pDf_list = [pdf1, pdf2, pdf3, pdf4, pdf5, pdf6, pdf7, pdf8]
  for i in range(num_pred_files):
    prediction_file = './' + str(i) + '_trip_type_prediction.csv'
    pDf_list[i] = pd.read_csv(prediction_file)

  ids = pDf_list[0]['VisitNumber']

  pAllDf = pd.concat(pDf_list, axis=1, keys=['pdf1', 'pdf2', 'pdf3', 'pdf4',  'pdf5', 'pdf6', 'pdf7', 'pdf8'])
  p_allDf = pAllDf.swaplevel(0, 1, axis=1).sortlevel(axis=1)

  p_allDf = p_allDf.groupby(level=0, axis=1).mean()

  p_allDf['VisitNumber'] = p_allDf['VisitNumber'].astype(np.int32)

  p_allDf.to_csv('./prediction_ensembled.csv', index=False, float_format='%.6f')
  print('ensembling done.')


if __name__ == "__main__":
  models = None
  n_parallel_models = 1
  mdl_type = 'xGB'
  models = begin_train(n_parallel_models, isPickled=False, loadPartial=False, model_type=mdl_type, n_est=200,
                        max_depth=None, cvPerformed=False, perform_model_test=True, t_split_size=0.1,
                        analyisPerformed=False, feature_selection_ON=True, vecPickled=True,
                        performGridSearch=False, performBlend=True, performPairWise=True, performChi=True,
                        testBiasVarience=True)
  begin_test(n_parallel_models, models, model_type=mdl_type, isPickled=False, loadPartial=False, vecPickled=False,
             performPairWise=False, performChi=False)
  ensemble_predictions(n_parallel_models)
  print('total program execution time(min) required: ', (time.time() - start_time) / 60)