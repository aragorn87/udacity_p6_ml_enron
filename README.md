# Udacity - Intro to Machine Learning - Identify Fraud from Enron Email
>  Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. 

The objective of this excercise is to identify the persons-of-interest within the enron email database who might have been involved in the fraud. We have with us a tidy dataset where information has been collated from the emails exchanged between the various employees as well as the financial information which was disclosed during the legal proceedings.

There are `146` data points (two of which are invalid entries) with `21` fields. Some of the key fields are:
-  Details on salary, bonus, stocks
-  number of emails sent, received
-  number of emails sent to and recieved from identified POIs
-  number of emails with shared receipts with POIs

Some of these fields (like salary, bonus, and stocks) give us a lot of information about how high up a particular individual was within the Enron's hierarchy. Information about the emails being exchanged with a POI also insinuates an individuals role in the scams/ frauds. 

The field of interest, `poi`,  is very skewed. The distribution is as follows:

| Column - poi | Count | Percentage |
| :----------- | ----- | ---------- |
| 1            | 18    | 12.5%      |
| 0            | 144   | 87.5%      |



> Were there any outliers in the data when you got it, and how did you handle those? 

The data had the entry `TOTAL` which is clearly a spreadsheet quirk. This was removed from the dataset. Also, an entry `THE TRAVEL AGENCY IN THE PARK` was also present which according the InsiderPay document states that 'Payments were made by Enron employees on account of business-related travel to The Travel Agency in the Park (later Alliance Worldwide), which was co-owned by the sister of Enron's former Chairman.  Payments made by the Debtor to reimburse employees for these expenses have not been included.'. Clearly, this particular entry is not an individual and hence was removed.

After removing these points, we still notice that the distribution is very uneven with a handful of individuals exhibiting very high values of salaries and bonuses. After careful consideration, there points were retained as these values are not outliers. 

![Outlier Treatment](/outlier.png?raw=true "Optional Title")

> What features did you end up using in your POI identifier, and what selection process did you use to pick them? 

The following features were incorporated in to the final model:
```python
    'fraction_poi_from',
    'fraction_shared_receipt',
    'deferred_income',
    'long_term_incentive',
    'bonus',
    'total_stock_value',
    'restricted_stock',
    'salary'
```

Features were selected using a combination of both logic as well as sklearn's GridSearch and pipeline functionality. For e.g. `exercised_stock_options` was omitted because it seems that there is a very high correlation between the `total_stock_value` and the `exercised_stock_options` (based on eyeballing the data). Further, the newly created feature `bonus_salary_ratio` was also omitted as both salary and bonus were already accounted for and taking the ratio would lead a a loss of information in cases where bonus was not null but salary was.

GridSearch was used to find the best `k` for `GaussianNB` model. The reason I used only GaussianNB is because this was an iterative process where I had already done an initial model testing (with gaussian, decision tree, SVC, and adaboost) and found that gaussian models were giving the best result. Although, what would have made more sense would be to have `SelectKBest` to feed into various other models as well to search for the best `k` (Topic for further research later!)

> Did you have to do any scaling? Why or why not? 

Initially, scaling was done using the `MinMaxScaler` as the units across all features vary a lot. However, given that we were not using any techniques (like kNN, or PCA) the scaling was essentially not of any importance. In the end, we did retain it but the scaling had very little impact on the model performance.

> As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it.

A couple of new features were added to the dataset as below:
-  `stock_to_payments = total_stock_value / total_payments` [vested interests dependent on the company performance indicator]
-  `bonus_salary_ratio = bonus / salary` [over and above the salary payments made to the individual]
-  `fraction_poi_to = from_poi_to_this_person / to_messages` [fraction of emails sent to the person from a POI]
-  `fraction_poi_from = from_this_person_to_poi / from_messages` [fraction of emails sent by the person to a POI]
-  `fraction_shared_receipt = shared_receipt_with_poi / to_messages` [fraction of received emails with a copied POI]
-  `poi_interaction_total = (from_poi_to_this_person + from_this_person_to_poi )/ (from_messages + to_messages)` [fraction of all interactions with a POI]

All the above features were created to check if they are any better in explaining the outcome variable than either the numerator or the denominator or both. As is displayed in the below mentioned scoring section, some of the features scored high than the original features. However, some of them were dropped as there was a loss of information due to the presence of `NaN` in either the numerator or the denominator.

> In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

`SelectKBest` was initially used to score all features. Based on the scores and intuition, some of the features were dropped. Later, using `k=8` (based on `GridSearchCV` output, as explained above), features were selected. The feature importance score matrix is as below:

| Feature                 | Score     | % NaNs    |
| ----------------------- | --------- | --------- |
| total_stock_value       | 24.182899 | 13.194444 |
| bonus                   | 20.792252 | 43.750000 |
| salary                  | 18.289684 | 34.722222 |
| fraction_poi_from       | 16.409713 | 40.277778 |
| deferred_income         | 11.458477 | 66.666667 |
| long_term_incentive     | 9.922186  | 54.861111 |
| restricted_stock        | 9.212811  | 24.305556 |
| fraction_shared_receipt | 9.101269  | 40.277778 |

> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

The following algorithms were used. The performance reported is based on a 30 percent test sample (single fold):

| Algorithm                  | Validation Precision Score | Validation Recall Score |
| -------------------------- | -------------------------- | ----------------------- |
| Gaussian Naive Bayes       | 0.5                        | 0.6                     |
| Decision Tree              | 0.125                      | 0.2                     |
| AdaBoost                   | 0.167                      | 0.2                     |
| Support Vector Classifiers | 0                          | 0                       |

Based on the above initial performance, `SVC` was dropped from the consideration set. `GridSearchCV` was performed on all the remaining algorithms while also increasing the number of folds to get a better sense of scores. (The above quoted scores for `GaussianNB` seem too good to be true which is usually the case when the samples are skewed. Cross Validation helps resolve such issues)

> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? 

Parameter fine tuning is a process of identifying the best parameters which give the best result for a given algorithm. It is an iterative process where the performance of models are compared by varying the key parameters. sklearn gives us the flexibility to automate the process using `GridSearchCV` where we can provide all possible options of a parameter we want to test out to find the best fit. 

Ignoring to fine tune the model could mean that we do not have an optimal solution. Also, we could be over fitting the data if we only design the model based on single set of parameters. 

Even though the selected algorithm was Gaussian Naive Bayes which doesn't have any parameters to fine tune, other algorithms were fine tuned as follows:

AdaBoost :
```python
  parameters = {'n_estimators': [50,100,120,150,200],
                'learning_rate': [0.1,0.5,1,1.5,2.0,2.5,5]}
  abc = AdaBoostClassifier()
  clf = GridSearchCV(abc, parameters, cv=10, scoring='f1')
```
Best fit model and scores:

```python
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
          n_estimators=50, random_state=None)
	Accuracy: 0.82636	Precision: 0.32379	Recall: 0.19800	F1: 0.24573	F2: 0.21468
	Total predictions: 14000	True positives:  396	False positives:  827	False negatives: 1604	True negatives: 11173
```

DecisionTree:

```python
parameters = {'criterion': ('gini','entropy'),
              'max_depth': [1,5,10,20],
              'min_samples_split': [2,4,6,8,10,20]}
dtc = DecisionTreeClassifier()
clf = GridSearchCV(dtc, parameters, cv=sss, scoring='f1')
```

Best fit model and scores:

```python
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=20,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=8, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
	Accuracy: 0.79493	Precision: 0.21994	Recall: 0.17100	F1: 0.19241	F2: 0.17896
	Total predictions: 14000	True positives:  342	False positives: 1213	False negatives: 1658	True negatives: 10787
```

GaussianNB:

Best fit model and scores:

```python
GaussianNB(priors=None)
	Accuracy: 0.85100	Precision: 0.47362	Recall: 0.38600	F1: 0.42534	F2: 0.40083
	Total predictions: 14000	True positives:  772	False positives:  858	False negatives: 1228	True negatives: 11142
```

>  What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis? 

Validation is the process of verifying the performance of a model over a 'hold-out' set. This practice tries to avoid the issue of over fitting where the model is very good at predicting the outcome when exposed to situations it has already been trained on. However, an over fit model fails to generalize and is often not the optimal solution to a problem.

In our case, validation was very important given the skewed classes within the data set. To avoid the case where the train and test samples were unevenly distributed, cross validation using startified sample shifts was used over 1000 folds. In this case, the training set is further segmented into two subsets which are randomly generated over 1000 iterations. The model is trained on the first (and the bigger) subset and then the performance is measured on the second subset. The performance of the model is averaged over 1000 runs to give a mean performance score of the evaluation metric being considered. This ensures that we avoid the pitfall of uneven distribution between train and test sets.

> Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

The evaluation metric which was primarily used to tune the models is f1 score which takes into consideration both the recall_score and the precision_score. Whereas we could have used either of the metrics (recall_score or precision_score) to fine tune the model, it wouldn't have been optimal as the intent of the excercise is to have good precision as well as recall.

Precision is a measure of how many times is an algorithm predicting the outcome correctly when it does predict a positive result. In context, if our model predicts POIs, precision score would quantify the number of times we would be correct in further investigating individuals.

Recall, on the other hand, is the fraction of times an algorithm is able to predict an outcome correctly. In context, if there are 100 POIs, an algorithm with a higher recall_score would be able to identify more of these individuals.

Accuracy score (% cases where classes were predicted correctly) is not a good measure in cases where a data set is skewed (very few POIs as compared to the total number of individuals). This is because, an algorithm which predicts all of them to be non-POIs will have a very high accuracy score. Such algorithms (which maximize accuracy_score) are of little use in cases where they might have further implications. For e.g. initiating an investigation against an otherwise non-POI person can have consequences (in terms of emotional impact on individuals, or a probable defamation case later on). Further, the bandwidth of the resources are also limited which put a constrain on the number of people that can be investigated.
