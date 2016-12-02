#!/usr/bin/python

## List of imports
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

######## Overview of dataset #######
"""
# Total no. of datapoints
print "total no. of datapoints: " + str(len(data_dict))

# how many POIs?
poi_count = 0
for name in data_dict.keys():
    if data_dict[name]['poi'] == 1:
        poi_count +=1
print "poi_count:" + str(poi_count)

# number of features?
print "number of features: " + str(len(data_dict["SKILLING JEFFREY K"]))

"""

####### Convert data_dict to pandas df & perform EDA #######
import numpy
import pandas
import matplotlib.pyplot as plt

df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))

# set the index of df to be the employees series:
df.set_index(employees, inplace=True)

# as dtypes are objects(strings), change values to floats (excl. poi, boolean)
df_new = df.apply(lambda x: pandas.to_numeric(x, errors='coerce')).copy()

# fill NaN values & plot features by poi/non-poi
"""
for ind in df_new:
    if ind != 'poi':
        print '\n', ind
        df_temp = df_new[['poi', ind]].fillna(0).copy()
        df_temp_grp = df_temp.groupby('poi')
        df_temp_grp.plot(kind='bar',alpha=0.75, rot=90, figsize=(20, 10))
        plt.rcParams.update({'font.size': 8})
        plt.show()
"""

# fill NaN values & calculate mean values by poi/non-poi
"""
for ind in df_new:
    if ind != 'poi':
        print '\n', ind
        df_temp = df_new[['poi', ind]].fillna(0).copy()
        print df_temp.groupby('poi').mean()
"""


####### Create new features, remove outliers, convert df back to df_dict #######

# 2 new finance features
df_new['bonus_to_salary_ratio'] = df_new.bonus.div(df_new.salary).fillna(0)
df_new['total_stock_value_to_salary_ratio'] = df_new.total_stock_value.div(df_new.salary).fillna(0)

# 2 new email features
df_new['fraction_to_poi'] = df_new.from_this_person_to_poi.div(df_new.from_messages).fillna(0)
df_new['fraction_from_poi'] = df_new.from_poi_to_this_person.div(df_new.to_messages).fillna(0)

# create a list of column names (new features list)
my_features_list = list(df_new.columns.values)
my_features_list.remove('poi')
my_features_list.insert(0, 'poi')

# print '\n dataframe features'
# print my_features_list

# replace np.nan with string 'NaN', create a dictionary from the dataframe
df_new = df_new.replace(numpy.nan,'NaN', regex=True)
df_dict = df_new.to_dict('index')

# Remove outliers
df_dict.pop('TOTAL', 0)
df_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

### Store to my_dataset for easy export below.
my_dataset = df_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



####### Test classifiers using simple train_test_split #######

# split data into train and test
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

NBclf = GaussianNB()
DTclf = DecisionTreeClassifier()
kNNclf = KNeighborsClassifier()

def test(clf):

    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    print recall_score(labels_test, pred)
    print precision_score(labels_test, pred)
    print clf.score(features_test, labels_test)

"""
print "NBclf"
print test(NBclf)
print "DTclf"
print test(DTclf)
print "kNNclf"
print test(kNNclf)
"""


####### Try a varity of classifiers & tune classifier to achieve
# better than .3 precision and recall using testing script. #######

# create instances
scaler = MinMaxScaler()
skb = SelectKBest()
nb = GaussianNB()
dt = DecisionTreeClassifier()

# create parameters for classifiers to be used in pipeline
params_nb = {
"SKB__k": range(2, 8),
}
params_dt = {
"SKB__k": range(2, 8),
"CLF__criterion": ["gini", "entropy"],
"CLF__min_samples_split": [5, 10],
"CLF__max_leaf_nodes": [10, 20, 30],
}

# tune classifier and obtain results based on parameters above
def get_results(classifier, parameters):
    steps = [('scaling',scaler), ('SKB', skb), ('CLF', classifier)]
    pipeline = Pipeline(steps)
    sss = StratifiedShuffleSplit(100, random_state = 7)

    ### use GridSearchCV to tune params & StratifiedShuffleSplit to cross validate
    gs = GridSearchCV(pipeline, parameters, cv=sss, scoring="f1")
    gs.fit(features, labels)

    ### create clf with optimal set of params
    clf = gs.best_estimator_
    ### create list with features selected by SelectKBest in GridSearchCV's optimal model
    feature_scores = ['%.2f' % elem for elem in clf.named_steps['SKB'].scores_]
    features_selected = [(my_features_list[i+1], feature_scores[i]) for i in clf.named_steps['SKB'].get_support(indices=True)]
    features_selected = sorted(features_selected, key=lambda feature: float(feature[1]), reverse=True)

    print '\n', '\n'
    print 'Here are the results for the classifier:', classifier
    print 'The features selected by SelectKBest (ranked by scores):'
    print features_selected, '\n'
    print "The best parameters: ", gs.best_params_, '\n'
    dump_classifier_and_data(clf, my_dataset, my_features_list)
    print "Tester Classification report: "
    test_classifier(clf, my_dataset, my_features_list)
    return clf

# get_results on classifiers
get_results(dt, params_dt)
get_results(nb, params_nb)
