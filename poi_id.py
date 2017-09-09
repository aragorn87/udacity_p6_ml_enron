
# coding: utf-8

# In[17]:

#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
outlier_check=['salary','bonus']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[18]:

my_data = data_dict
data = featureFormat(my_data, outlier_check, sort_keys = True)

### Task 2: Remove outliers
import matplotlib.pyplot

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



# In[19]:

my_data.pop('TOTAL',0)
my_data.pop('THE TRAVEL AGENCY IN THE PARK',0)


# In[20]:

data = featureFormat(my_data, outlier_check, sort_keys = True)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# In[21]:

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_data = data_dict
for key, item in my_data.iteritems():
    #stock_to_payments
    if ((my_data[key]['total_payments']=='NaN') or (my_data[key]['total_payments']==0) or (my_data[key]['total_stock_value']=='NaN')):
        my_data[key]['stock_to_payments']='NaN'
    else:
        my_data[key]['stock_to_payments']=1.0*my_data[key]['total_stock_value']/my_data[key]['total_payments']
    
    #bonus_salary_ratio
    if ((my_data[key]['bonus']=='NaN') or (my_data[key]['salary']=='NaN') or (my_data[key]['salary']==0)):
        my_data[key]['bonus_salary_ratio'] = 'NaN'
    else:
        my_data[key]['bonus_salary_ratio'] = 1.0* my_data[key]['bonus'] / my_data[key]['salary']
    
    #fraction_poi_to
    if ((my_data[key]['from_poi_to_this_person']=='NaN') or (my_data[key]['to_messages']=='NaN') or (my_data[key]['to_messages']==0)):
        my_data[key]['fraction_poi_to'] = 'NaN'
    else:
        my_data[key]['fraction_poi_to'] = 1.0* my_data[key]['from_poi_to_this_person'] / my_data[key]['to_messages']
    
    #fraction_poi_from
    if ((my_data[key]['from_this_person_to_poi']=='NaN') or (my_data[key]['from_messages']=='NaN') or (my_data[key]['from_messages']==0)):
        my_data[key]['fraction_poi_from'] = 'NaN'
    else:
        my_data[key]['fraction_poi_from'] = 1.0* my_data[key]['from_this_person_to_poi'] / my_data[key]['from_messages']
    
    #fraction_shared_receipt
    if ((my_data[key]['shared_receipt_with_poi']=='NaN') or (my_data[key]['to_messages']=='NaN') or (my_data[key]['to_messages']==0)):
        my_data[key]['fraction_shared_receipt'] = 'NaN'
    else:
        my_data[key]['fraction_shared_receipt'] = 1.0* my_data[key]['shared_receipt_with_poi'] / my_data[key]['to_messages']
    
    #poi_interation_total
    if ((my_data[key]['from_poi_to_this_person']=='NaN') or (my_data[key]['from_this_person_to_poi']=='NaN') or (my_data[key]['from_messages']=='NaN') or (my_data[key]['to_messages']=='NaN') or (my_data[key]['from_messages']+my_data[key]['to_messages']==0)):
        my_data[key]['poi_interaction_total'] = 'NaN'
    else:
        my_data[key]['poi_interaction_total'] = 1.0*(my_data[key]['from_poi_to_this_person']+my_data[key]['from_this_person_to_poi'])/(my_data[key]['from_messages']+my_data[key]['to_messages'])


# In[22]:

features_list=['poi', 'bonus', 'bonus_salary_ratio', 'deferral_payments','deferred_income', 'director_fees','exercised_stock_options','expenses','fraction_poi_from','fraction_poi_to','fraction_shared_receipt','from_messages','from_poi_to_this_person','from_this_person_to_poi','loan_advances','long_term_incentive','other','poi_interaction_total','restricted_stock','restricted_stock_deferred','salary','shared_receipt_with_poi','stock_to_payments','to_messages','total_payments','total_stock_value']
#features_list=['poi', 'bonus', 'bonus_salary_ratio', 'deferral_payments','deferred_income', 'director_fees','expenses','fraction_poi_from','fraction_poi_to','fraction_shared_receipt','from_messages','from_poi_to_this_person','from_this_person_to_poi','loan_advances','long_term_incentive','other','poi_interaction_total','restricted_stock','restricted_stock_deferred','salary','shared_receipt_with_poi','stock_to_payments','to_messages','total_payments','total_stock_value']

### Extract features and labels from dataset for local testing
data = featureFormat(my_data, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#scaling 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features)

features = scaler.transform(features)

# finding k best features
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k='all')
selector.fit(features, labels)

feature_importance_table=pd.DataFrame(columns=['features','importance'])
i=0
feature_importance_table['features'] = features_list[1:]
feature_importance_table['importance'] = selector.scores_
#for n,g in zip(features_list[1:],selector.scores_):
#    feature_importance_table['feature'].append(n)
#    feature_importance_table['importance'].append(g)
    
#print selector.scores_

print feature_importance_table.sort('importance', ascending=False)
#sort_values(by=(['importance']), axis=1, ascending=False)
print sum(feature_importance_table['importance'])


# In[39]:

#removin excersides_stock-options as it is equivalent to total_stock_value
#removing bosus salayr ration as information loss can happen


# In[23]:

#features_list=['poi', 'bonus', 'bonus_salary_ratio', 'deferral_payments','deferred_income', 'director_fees','exercised_stock_options','expenses','fraction_poi_from','fraction_poi_to','fraction_shared_receipt','from_messages','from_poi_to_this_person','from_this_person_to_poi','loan_advances','long_term_incentive','other','poi_interaction_total','restricted_stock','restricted_stock_deferred','salary','shared_receipt_with_poi','stock_to_payments','to_messages','total_payments','total_stock_value']
features_list=['poi', 'bonus','deferral_payments','deferred_income', 'director_fees','expenses','fraction_poi_from','fraction_poi_to','fraction_shared_receipt','from_messages','from_poi_to_this_person','from_this_person_to_poi','loan_advances','long_term_incentive','other','poi_interaction_total','restricted_stock','restricted_stock_deferred','salary','shared_receipt_with_poi','stock_to_payments','to_messages','total_payments','total_stock_value']

### Extract features and labels from dataset for local testing
data = featureFormat(my_data, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k='all')
selector.fit(features, labels)

feature_importance_table=pd.DataFrame(columns=['features','importance', 'missing_values'])
i=0
feature_importance_table['features'] = features_list[1:]
feature_importance_table['importance'] = selector.scores_
#for n,g in zip(features_list[1:],selector.scores_):
#    feature_importance_table['feature'].append(n)
#    feature_importance_table['importance'].append(g)
    
#print selector.scores_


#sort_values(by=(['importance']), axis=1, ascending=False)
#print sum(feature_importance_table['importance'])

#for key, item in my_data.iteritems():
#    for key2, item2 in item.iteritems():
total_count= len(my_data)
missing_val=[]
for i, series in feature_importance_table.iterrows():
    count_nan=0
    for key, item in my_data.iteritems():
        if item[series['features']]== 'NaN':
            count_nan +=1
    per_nan = 100.0*count_nan/total_count
    missing_val.append(per_nan)
feature_importance_table['missing_values'] = missing_val
#next(df.iterrows())[1]
print feature_importance_table.sort('importance', ascending=False)
    


# In[24]:

#selecting 10 best features
features_list=['poi', 'total_stock_value','bonus','salary', 'fraction_poi_from','deferred_income','long_term_incentive','restricted_stock','fraction_shared_receipt','total_payments','shared_receipt_with_poi']
selector = SelectKBest(k=10)
selector.fit(features, labels)

features = selector.transform(features)


# In[25]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.metrics import recall_score, precision_score

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
#print "GaussianNB accuracy: " , recall_score(labels_test, clf.predict(features_test)), precision_score(labels_test, clf.predict(features_test))


from sklearn.svm import SVC
clf = SVC()
clf.fit(features_train, labels_train)
#print "SVM accuracy: " , recall_score(labels_test, clf.predict(features_test)), precision_score(labels_test, clf.predict(features_test))

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
#print "Decision Tree accuracy: " , recall_score(labels_test, clf.predict(features_test)), precision_score(labels_test, clf.predict(features_test))

from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier()
clf.fit(features_train, labels_train)
#print "Adaboost accuracy: " , recall_score(labels_test, clf.predict(features_test)), precision_score(labels_test, clf.predict(features_test))





# In[26]:

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

#From the above excercise, it seems that out GaussianNB is doing a good job. However, it is possible
#that the performance is due to the uneven distribution of samples between training and test sets. Thererfore,
#we will now use StratifiedShiffleSplit over 1000 folds to come up with the best model and chech the score using 
#testet.py file

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from tester import test_classifier

sss = StratifiedShuffleSplit(1000, random_state = 42)

## Adaboost Classifier - fine tuning
parameters = {'n_estimators': [50,100,120,150,200],
             'learning_rate': [0.1,0.5,1,1.5,2.0,2.5,5]}
abc = AdaBoostClassifier()
clf = GridSearchCV(abc, parameters, cv=10, scoring='f1')
clf.fit(features_train, labels_train)
clf=clf.best_estimator_
test_classifier(clf, my_data, features_list, folds = 1000)


## DecisionTree Classifier - fine tuning
parameters = {'criterion': ('gini','entropy'),
              'max_depth': [1,5,10,20],
             'min_samples_split': [2,4,6,8,10,20]}
dtc = DecisionTreeClassifier()

clf = GridSearchCV(dtc, parameters, cv=sss, scoring='f1')
clf.fit(features_train, labels_train)
clf=clf.best_estimator_
test_classifier(clf, my_data, features_list, folds = 1000)

##GaussianNB CV - nothing to fine tune

parameters = {}
gnb = GaussianNB()
clf = GridSearchCV(gnb, parameters, cv=sss, scoring='f1')
clf.fit(features_train, labels_train)
clf= clf.best_estimator_
test_classifier(clf, my_data, features_list, folds = 1000)

##based on the performance, the selected model is GaussianNB


# In[40]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
my_dataset=my_data
dump_classifier_and_data(clf, my_dataset, features_list)

