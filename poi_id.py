# coding: utf-8
#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot





### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
outlier_check=['salary','bonus']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
my_data = data_dict
data = featureFormat(my_data, outlier_check, sort_keys = True)

### Task 2: Remove outliers


for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

my_data.pop('TOTAL',0)
my_data.pop('THE TRAVEL AGENCY IN THE PARK',0)

data = featureFormat(my_data, outlier_check, sort_keys = True)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


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

### Generating a list of features for future reference; removing 'email_address' and moving 'poi' to the front of the list
features_list=[]
features_list=my_data['METTS MARK'].keys()
features_list.remove('email_address')
features_list.remove('poi')
features_list=['poi']+features_list


### Extract features and labels from dataset for local testing
data = featureFormat(my_data, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#scaling 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features)

features = scaler.transform(features)

# finding feature importance score of all features
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k='all')
selector.fit(features, labels)

feature_importance_table=pd.DataFrame(columns=['features','importance'])
feature_importance_table['features'] = features_list[1:]
feature_importance_table['importance'] = selector.scores_

print feature_importance_table.sort('importance', ascending=False)
print sum(feature_importance_table['importance'])


#removing exercised_stock-options as it is equivalent to total_stock_value
#removing bonus salary ratio as information loss can happen
features_list.remove('exercised_stock_options')
features_list.remove('bonus_salary_ratio')

### Extract features and labels from dataset for local testing
data = featureFormat(my_data, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Feature selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
select = SelectKBest()
clf = GaussianNB()
steps = [('feature_selection', select),
        ('gaussian_nb', clf)]
pipeline = Pipeline(steps)
parameters = dict(feature_selection__k=range(2,len(features_list)-1))
cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1')
cv.fit(features, labels )
print cv.best_score_, cv.best_params_['feature_selection__k']

#selecting k best features, based on above output
select = SelectKBest(k=cv.best_params_['feature_selection__k'])
select.fit(features, labels)
features = select.transform(features)

#making corresponding changes to features_list
m=features_list[1:]
x=zip(select.get_support(), m)
features_list=[i[1] for i in x if i[0]==True]
features_list=['poi']+features_list

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
data = featureFormat(my_data, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.metrics import recall_score, precision_score

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.3, random_state=42)

clf = GaussianNB()
clf.fit(features_train, labels_train)
print "GaussianNB accuracy: " , recall_score(labels_test, clf.predict(features_test)), precision_score(labels_test, clf.predict(features_test))


from sklearn.svm import SVC
clf = SVC()
clf.fit(features_train, labels_train)
print "SVM accuracy: " , recall_score(labels_test, clf.predict(features_test)), precision_score(labels_test, clf.predict(features_test))

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print "Decision Tree accuracy: " , recall_score(labels_test, clf.predict(features_test)), precision_score(labels_test, clf.predict(features_test))

from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier()
clf.fit(features_train, labels_train)
print "Adaboost accuracy: " , recall_score(labels_test, clf.predict(features_test)), precision_score(labels_test, clf.predict(features_test))

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

#From the above exercise, it seems that out GaussianNB and DecisionTree are doing a good job. However, it is possible
#that the performance is due to the uneven distribution of samples between training and test sets. Therefore,
#we will now use StratifiedShiffleSplit over 1000 folds to come up with the best model and check the score using 
#tester.py file

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
