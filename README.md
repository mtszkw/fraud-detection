# Credit Card Fraud Detection

This code compares usage of logistic regression and LightGBM classifier to determine whether a transaction is a [credit card fraud](https://en.wikipedia.org/wiki/Credit_card_fraud). Whole dataset is available on Kaggle ([here](https://www.kaggle.com/mlg-ulb/creditcardfraud)) and contains transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

To reduce data imbalance, I choose a sample from data so that fraud/non-fraud transactions are in even proportion.

Using logistic regression and LightGBM models with cross-validation (5 folds) leads to the highest F1 score around **97,7%** on the randomly extracted test set. Perhaps further preprocessing (avoiding information loss from undersampling) or even more parameter tuning could improve that score a bit.

Kaggle kernel: https://www.kaggle.com/mtszkw/lightgbm-vs-logistic-regression-f1-97-7
