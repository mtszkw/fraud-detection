# Credit Card Fraud Detection

This code compares usage of sklearn's `LogisticRegression` and `LightGBM` classifiers to determine whether a transaction is a [credit card fraud](https://en.wikipedia.org/wiki/Credit_card_fraud). Full dataset is [available on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and contains a lists of transactions that occurred in two days. The dataset is highly unbalanced as it contains 492 frauds (0.172%) out of 284,807 transactions.

### Balancing data
This notebook is just a quick draft, so to reduce imbalance a sample from data is selected so that fraud/non-fraud transactions are in even proportion. Most likely, using more sophisticated imbalance reduction technique would improve the final score.

### Score
Using Logistic Regression and LightGBM models with cross-validation (5 folds) leads to the highest F1 score around **97,7%** on the randomly extracted test set (20% of data). There is still a lot to improve: tesitng other models, more exhaustive hyperparameter tuning and proper class balancing might all improve accuracy of classification.

Kaggle kernel: https://www.kaggle.com/mtszkw/lightgbm-vs-logistic-regression-f1-97-7
