# XGBoost_stock_prediction
XGBoost is known to be fast and achieve good prediction results as compared to the regular gradient boosting libraries. This project attempts to predict stock price direction by using the stock's daily data and indicators derived from its daily data as predictors. A classification solution.

## Results

[image1]: https://github.com/jiewwantan/XGBoost_stock_prediction/blob/master/features_histograms.png "Feature data histograms"
![Feature data histograms][image1]

## Improvement suggestion
Before arriving at XGboostCV, GridsearchCV (all hyperparameters tuning at once) and XGboosting (one hyperparameter tuning at a time) were tried. The former took a long time to train and achieve lacklustre result (below 0.7 accuracy), the latter performs much faster but is seriously overtrained. Even if the current result doesn't overfit, the performance ~ 0.7 test accuracy is lacklustre, given the number of features to learn from. I suspect this can be due to autocorrelation and autoregressive nature of the time series data and that slicing the data at the wrong place diconnects its learnability. It may be necessary to combine with other models, such as econometric model and other non-linear model to learn well from time-series stock data.

## References
Python API Reference, https://xgboost.readthedocs.io/en/latest/tutorials/model.html
Complete Guide to Parameter Tuning in XGBoost, https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
Strategy: XGBoost Classification + Technical Indicators, http://allenfrostline.com/2017/03/12/strategy-xgboost-classification-technical-indicators/
