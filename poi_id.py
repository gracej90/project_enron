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

# as dtypes are objects(strings), change values floats (excl. poi, boolean)
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

# print df_new['bonus_to_salary_ratio']

# create a list of column names (new features list)
new_features_list = list(df_new.columns.values)
new_features_list.remove('poi')
new_features_list.insert(0, 'poi')

# print '\n dataframe features'
# print new_features_list

# replace np.nan with string 'NaN', create a dictionary from the dataframe
df_new = df_new.replace(numpy.nan,'NaN', regex=True)
df_dict = df_new.to_dict('index')

# Remove outliers
df_dict.pop('TOTAL', 0)
df_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

### Store to my_dataset for easy export below.
my_dataset = df_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
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
# better than .3 precision and recall using our testing script. #######

### create gs using pipeline with scaling, feature selection, classification

# create instances
scaler = MinMaxScaler()
skb = SelectKBest()
pca = PCA()
GaussianNB = GaussianNB()
tree = DecisionTreeClassifier()

# create classifier using pipeline & pass to GridSearchCV
pipe_NB = Pipeline(steps=[('scaling', scaler), ('skb', skb), ('nb', GaussianNB)])
# pipe_tree = Pipeline(steps=[('scaling', scaler), ('pca', pca), ('tree', tree)])

# create parameters
param_grid = {
"skb__k": [2,3,4,5,6,7,8],
#"pca__n_components": range(2,10),
#"tree__criterion": ["gini", "entropy"],
#"tree__min_samples_split": [5, 10],
#"tree__max_leaf_nodes": [10, 20, 30],
}

### use GridSearchCV to tune params & StratifiedShuffleSplit to cross validate
sss = StratifiedShuffleSplit(100, random_state=7)
gs = GridSearchCV(pipe_NB, param_grid, scoring = 'f1', cv = sss)

### fit classifier
gs.fit(features, labels)

"""
print 'Best score: %0.3f' % gs.best_score_
print 'Best parameters set:'
best_parameters = gs.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
       print '\t%s: %r' % (param_name, best_parameters[param_name])
"""

### create clf with optimal set of params
clf = gs.best_estimator_

# create a new list that contains the features selected by SelectKBest
# in the optimal model selected by GridSearchCV.
K_best = clf.named_steps['skb']

feature_scores = ['%.2f' % elem for elem in K_best.scores_]
features_selected =[(new_features_list[i+1], feature_scores[i]) for i in K_best.get_support(indices=True)]
features_selected = sorted(features_selected, key=lambda feature: float(feature[1]) , reverse=True)

"""
print 'Features selected by SelectKBest ranked by score: '
print features_selected
"""

####### Dump your classifier, dataset, and features_list so anyone can check your results. #######

dump_classifier_and_data(clf, my_dataset, new_features_list)

# check results of classifier
from tester import test_classifier
print "Tester Classification report"
test_classifier(clf, my_dataset, new_features_list)
