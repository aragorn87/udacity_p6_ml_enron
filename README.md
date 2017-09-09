# udacity_p6_ml_enron

>  Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. 

The objective of this excercise is to identify the persons-of-interest within the enron email database who might have been involved in the fraud. We have with us a tidy dataset where information has been collated from the emails exchanged between the various employees as well as the financial information which was disclosed during the legal proceedings.

There are 146 data points (two of which are invalid entries) with 21 fields. Some of the key fields are:
-  Details on salary, bonus, stocks
-  number of emails sent, received
-  number of emails sent to and recieved from identified POIs
-  number of emails with shared receipts with POIs

Some of these fields (like salary, bonus, and stocks) give us a lot of information about how high up a particular individual was within the Enron's hierarchy. Information about the emails being exchanged with a POI also insinuates an individuals role in the scams/ frauds. 

> Were there any outliers in the data when you got it, and how did you handle those? 

The data had the entry 'TOTAL' which is clearly a spreadsheet quirk. This was removed from the dataset. Also, an entry 'THE TRAVEL AGENCY IN THE PARK' was also present which according the InsiderPay document states that 'Payments were made by Enron employees on account of business-related travel to The Travel Agency in the Park (later Alliance Worldwide), which was coowned by the sister of Enron's former Chairman.  Payments made by the Debtor to reimburse employees for these expenses have not been included.'. Clearly, this particular entry is not an individual and hence was removed.

After removing these points, we still notice that the distribution is very uneven with a handful of individuals exhibiting very high values of salaries and bonuses. After careful consideration, there points were retained as these values are not outliers. 

![Outlier Treatment](/outlier.png?raw=true "Optional Title")

> What features did you end up using in your POI identifier, and what selection process did you use to pick them? 

The following features were incorporated in to the final model:
-  poi
-  total_stock_value
-  bonus
-  salary
-  fraction_poi_from
-  deferred_income
-  long_term_incentive
-  restricted_stock
-  fraction_shared_receipt
-  total_payments
-  shared_receipt_with_poi

While SelectKBest was used to find the best 10 features, some of the features were manually eliminated on the basis of understanding of the data. For e.g. 'excercised_stock_options' was emitted evon though it had a very high score in the SelectKBest routine. This was done because it seems that there is a very high correlation between the total_stock_value and the excercised_stock_options (based on eyeballing the data). Further, the newly created feature bonus_salary_ratio was also omitted as both salary and bonus were already accounted for and taking the ratio would lead a a loss of information in cases where bonus was not null but salary was.

> Did you have to do any scaling? Why or why not? 
Initially, scaling was done using the MinMaxScaler as the units across all features vary a lot. However, given that we were not using any techniques (like kNN, or PCA) the scaling was essentially not of any importance. In the end, we did retain it but the scaling had very little impact on the model performance.

> As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it.

A couple of new features were added to the dataset as below:
-  stock_to_payments = total_stock_value / total_payments [vested interests dependent on the company performance indicator]
-  bonus_salary_ratio = bonus / salary [over and above the salary payments made to the individual]
-  fraction_poi_to = from_poi_to_this_person / to_messages [fraction of emails sent to the person from a POI]
-  fraction_poi_from = from_this_person_to_poi / from_messages [fraction of emails sent by the person to a POI]
-  fraction_shared_receipt = shared_receipt_with_poi / to_messages [fraction os recieved emails with a copied POI]
-  poi_interation_total = (from_poi_to_this_person + from_this_person_to_poi )/ (from_messages + to_messages) [fraction of all interactions with a POI]

All the above features were created to check if they are any better in explaining the outcome variable than either the numerator or the denominator or both. As is desplayed in the below mentioned scoring section, some of the features scored high than the original features. However, some of them were dropped as there was a loss of information due to the presence of NaNs in either the numerator or the denominator.

> In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

SelectKBest was initially used to score all features. Based on the scores and intuition, some of the features were dropped. Later, using k=10, features were selected. The fetaure importance score matrix is as below:

| Feature | Score | % NaNs |
| --- | --- | ---|
| exercised_stock_options | 24.815080 | 29.861111 |
| total_stock_value | 24.182899 | 13.194444 |
| bonus | 20.792252 | 43.750000 |
| salary | 18.289684 | 34.722222 |
| fraction_poi_from | 16.409713 | 40.277778 |
| deferred_income | 11.458477 | 66.666667 |
| bonus_salary_ratio | 10.783585 | 43.750000 |
| long_term_incentive | 9.922186 | 54.861111 |
| restricted_stock | 9.212811 | 24.305556 |
| fraction_shared_receipt | 9.101269 | 40.277778 |
| total_payments | 8.772778 | 14.583333 |
| shared_receipt_with_poi | 8.589421 | 40.277778 |
| loan_advances | 7.184056 | 97.916667 |
| expenses | 6.094173 | 34.722222 |
| poi_interaction_total | 5.399370 | 40.277778 |
| from_poi_to_this_person | 5.243450 | 40.277778 |
| other | 4.187478 | 36.805556 |
| fraction_poi_to | 3.128092 | 40.277778 |
| from_this_person_to_poi | 2.382612 | 40.277778 |
| director_fees | 2.126328 | 88.888889 |
| to_messages | 1.646341 | 40.277778 |
| deferral_payments | 0.224611 | 73.611111 |
| from_messages | 0.169701 | 40.277778 |
| restricted_stock_deferred | 0.065500 | 88.194444 |
| stock_to_payments | 0.022819 | 25.694444 |

> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

The following algorithms were used. The performance reported is based on a 30 percent test sample (single fold):

| Algorithm | Validation Precision Score | Validation Recall Score |
| --- | --- | --- |
| Gaussian Naive Bayes | 0.5 | 0.6 |
| Decision Tree | 0.125 | 0.2 |
| AdaBoost | 0.167 | 0.2 |
| Support Vector Classifiers| 0 | 0 |

Based on the above inital perfromance, SVC was dropped from the consideration set. GridSearchCV was performed on all the remaining algorithms while also increasing the number of folds to get a better sense of scores. (The above quoted scores for GaussianNB seem too good to be true which is usually the case when the samples are skewed. Cross Validation helps resolve such issues)

> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? 

Parameter fine tuning is a process of identifying the best parameters which give the best result for a given algortithm. It is an iterative process where the performance of models are compared by varying the key parameters. sklearn gives us the flexibility to automate the process using GridSearchCV where we can provide all possible options of a parameter we want to test out to find the best fit. 

Ignoring to fine tune the model could mean that we do not have an optimal solution. Also, we could be overfitting the data if we only design the model based on single set of parameters. 

Even though the selected algorithm was Gaussian Naive Bayes which doesnt have any parameters to fine tune, other algorithms were fine tuned as follows:

AdaBoost :
parameters = {'n_estimators': [50,100,120,150,200],
             'learning_rate': [0.1,0.5,1,1.5,2.0,2.5,5]}
	     
abc = AdaBoostClassifier()

clf = GridSearchCV(abc, parameters, cv=10, scoring='f1')

Best fit model and scores:

AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=2.0, n_estimators=100, random_state=None)
	Accuracy: 0.78513	Precision: 0.25333	Recall: 0.31400	F1: 0.28042	F2: 0.29965
	Total predictions: 15000	True positives:  628	False positives: 1851	False negatives: 1372	True negatives: 11149
  
DecisionTree:
parameters = {'criterion': ('gini','entropy'),
              'max_depth': [1,5,10,20],
             'min_samples_split': [2,4,6,8,10,20]}
             
dtc = DecisionTreeClassifier()

clf = GridSearchCV(dtc, parameters, cv=sss, scoring='f1')

Best fit model and scores:

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
	Accuracy: 0.79793	Precision: 0.22389	Recall: 0.20900	F1: 0.21619	F2: 0.21182
	Total predictions: 15000	True positives:  418	False positives: 1449	False negatives: 1582	True negatives: 11551
  
GaussianNB:

Best fit model and scores:

GaussianNB(priors=None)
	Accuracy: 0.83840	Precision: 0.37198	Recall: 0.30800	F1: 0.33698	F2: 0.31897
	Total predictions: 15000	True positives:  616	False positives: 1040	False negatives: 1384	True negatives: 11960

>  What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis? 

Validation is the process of verifying the performance of a model over a 'hold-out' set. This practice tries to avoid the issue of overfitting where the model is very good at predicting the outcome when exposed to situations it has already been trained on. However, an overfit model fails to generalize and is often not the optimal solution to a problem.

In our case, validation was very important given the skewed classes within the data set. To avoid the case where the train and test samples were unevenly distributed, cross validation using startified sample shifts was used over 1000 folds. In this case, the training set is further segmented into two subsets which are randomly generated over 1000 iterations. The model is trained on the first (and the bigger) subset and then the performance is measured on the second subset. The performance of the model is averaged over 1000 runs to give a mean performance score of the evaluation metric being considered. This ensures that we avoid the pitfall of uneven distribution between train and test sets.



> Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

The evaluation metric which was primarily used to tune the models is f1 score which takes into consideration both the recall_score and the precision_score. Whereas we could have used either of the metrics (recall_score or precision_score) to fine tune the model, it wouldn't have been optimal as the intent of the excercise is to have good precision as well as recall.

Precision is a measure of how many times is an algorithm predictive the outcome correctly when it does predict a positive result. In context, if our model predicts that POIs, precision score would quantify the number of times we would be correct in further investigating individuals.

Recall, on the other hand, is the fraction of times an algorithm is able to predict an outcome correctly. In context, if there are 100 POIs, an algorithm with a higher recall_score would be able to identify more of these individuals.

Accuracy score (% cases where classes were predicted correctly) is not a good measure in cases where a data set is skewed (very few POIs as compared to the total number of individuals). This is because, an algorithm which predicts all of them to be non-POIs will have a very high accuracy score. Such algorithms (which maximize accuracy_score) are of little use in cases where they might have further implications. For e.g. initiating an investigation against an otherwise non-POI person can have consequences (in terms of emotional impact on individuals, or a probable defamation case later on). Further, the bandwidth of the resources are also limited which put a constrain on the number of people that can be investigated.
