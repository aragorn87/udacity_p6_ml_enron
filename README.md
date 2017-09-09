# udacity_p6_ml_enron

## 1. 
>  Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those? 
The objective of this excercise is to identify the persons-of-interest within the enron email database who might have been involved in the fraud. We have with us a tidy dataset where information has been collated from the emails exchanged between the various employees as well as the financial information which was disclosed during the legal proceedings.

There are 146 data points (two of which are invalid entries) with 21 fields. Some of the key fields are:
-  Details on salary, bonus, stocks
-  number of emails sent, received
-  number of emails sent to and recieved from identified POIs
-  number of emails with shared receipts with POIs

Some of these fields (like salary, bonus, and stocks) give us a lot of information about how high up a particular individual was within the Enron's hierarchy. Information about the emails being exchanged with a POI also insinuates an individuals role in the scams/ frauds. 

Outliers: The data had the entry 'TOTAL' which is clearly a spreadsheet quirk. This was removed from the dataset. Also, an entry 'THE TRAVEL AGENCY IN THE PARK' was also present which according the InsiderPay document states that 'Payments were made by Enron employees on account of business-related travel to The Travel Agency in the Park (later Alliance Worldwide), which was coowned by the sister of Enron's former Chairman.  Payments made by the Debtor to reimburse employees for these expenses have not been included.'. Clearly, this particular entry is not an individual and hence was removed.

After removing these points, we still notice that the distribution is very uneven with a handful of individuals exhibiting very high values of salaries and bonuses. After careful consideration, there points were retained as these values are not outliers. 

![Outlier Treatment](/outlier.png?raw=true "Optional Title")

## 2.
What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

## 3.
What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

## 4.
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

## 5.
What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

## 6.
Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]
