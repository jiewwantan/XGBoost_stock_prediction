# XGBoost_stock_prediction
XGBoost is known to be fast and achieve good prediction results as compared to the regular gradient boosting libraries. This project attempts to predict stock price direction by using the stock's daily data and indicators derived from its daily data as predictors. A classification solution.

## Data Investigation & Preprocessing

### The histogram illustrates the richness of data
[image1]: https://github.com/jiewwantan/XGBoost_stock_prediction/blob/master/features_histograms.png "Feature data histograms"
![Feature data histograms][image1]

### The correlation of the feature data between each other
[image2]: https://github.com/jiewwantan/XGBoost_stock_prediction/blob/master/plot_corr_heatmap.png "Feature data correlation heatmap"
![Feature data correlation heatmap][image2]

### The importance of each feature data according to the Feature Selection algorithm
[image3]: https://github.com/jiewwantan/XGBoost_stock_prediction/blob/master/plot_importance.png "Feature data importance"
![Feature data importance][image3]

### The correlation of the feature data between each other after feature selection
[image4]: https://github.com/jiewwantan/XGBoost_stock_prediction/blob/master/plot_corr_heatmap_fs.png "Selected Feature data correlation heatmap"
![Selected Feature data correlation heatmap][image4]

## Results

### Training accuracy
[image5]: https://github.com/jiewwantan/XGBoost_stock_prediction/blob/master/training_auc.png "Training Accuracy"
![Training Accuracy][image5]

### Training loss
[image6]: https://github.com/jiewwantan/XGBoost_stock_prediction/blob/master/training_logloss.png "Training Loss"
![Training Loss][image6]

## Improvement suggestion
Before arriving at XGboostCV, GridsearchCV (all hyperparameters tuning at once) and XGboosting (one hyperparameter tuning at a time) were tried. The former took a long time to train and achieve lacklustre result (below 0.7 accuracy), the latter performs much faster but is seriously overtrained. Even if the current result doesn't overfit, the performance ~ 0.7 test accuracy is lacklustre, given the number of features to learn from. I suspect this can be due to autocorrelation and autoregressive nature of the time series data and that slicing the data at the wrong place diconnects its learnability. It may be necessary to combine with other models, such as econometric model and other non-linear model to learn well from time-series stock data.
