# --------------------------------------- IMPORT LIBRARIES -------------------------------------------
# The following warning codes is to suppress sklearn's forced deprecation warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import talib as tb
import numpy as np
import sys
import numpy as np
import seaborn as sns
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as pdr
import fix_yahoo_finance as yf
import xgboost
from xgboost import XGBClassifier
from time import sleep
from datetime import datetime as dt
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.model_selection import train_test_split
from pprint import pprint
# Libraries required by FeatureSelector()
import lightgbm as lgb
import gc
from itertools import chain
# --------------------------------------- GLOBAL PARAMETERS -------------------------------------------
# Range of date to train and predict
START = datetime(2008, 9, 1)
END = datetime(2018, 10, 4)
# ------------------------------------------------ CLASSES --------------------------------------------
class UserInput:
    """
    The class to contain user input function.

    Returns:
        symbol: stock symbol entered by user

    Raises:
        NameError:: When the symbol user entered is not a valid symbol.
        ValueError: When no or not enough historical data from the source.
    """

    @staticmethod
    def get_symbol():
        """
        This function gets user to enter a stock symbol.
        Exceptions handlers are in place to ensure user enter a valid stock symbol.
        """

        validity = False
        while validity is False:
            try:
                symbol = input("Please enter a NYSE or NASDAQ stock symbol > \b")
                # Make all alphabets uppercase
                symbol = symbol.upper()

                user_confirm = []
                # If user input is not within the expected answers or user just hit enter without entering value
                while user_confirm not in ['n', 'N', 'no', 'No', 'NO', 'y', 'Y', 'yes', 'Yes', 'YES'] and symbol != "":

                    # Get user to confirm his/her input
                    user_confirm = input("Stock quote: [ %s ] is received, enter y/n to confirm >" % symbol)

                    # If user says No
                    if user_confirm in ['n', 'N', 'no', 'No', 'NO']:
                        pass

                    # If user says Yes
                    elif user_confirm in ['y', 'Y', 'yes', 'Yes', 'YES']:
                        print ("Please wait, checking stock symbol's validity ...")
                        try:
                            # Check if data is available for this stock
                            daily_data = pdr.get_data_yahoo(symbol, START, END)
                        except:
                            pass
                        if len(daily_data) > 2520:
                            print ("Great, you have entered a valid stock symbol: {}".format(symbol))
                            validity = True
                        else:
                            validity = False
                            raise ValueError

                    # If user input is not within the expected answers, re-loop and prompt user input again
                    else:
                        pass

            # When stock symbol is not recognized by NASDAQ, chances are it is not a valid stock symbol
            except:
                print('Entry is not a valid stock symbol.')
        return symbol

class Data:

    def __init__(self, symbol):
        self.q = symbol
        self._get_daily_data()
        self.technical_indicators_df()

    def _get_daily_data(self):
        """
        This class prepares data by downloading historical data from Yahoo Finance,

        """
        flag = False
        # Set counter for download trial
        counter = 0

        # Safety loop to handle unstable Yahoo finance download
        while not flag and counter < 6:
            try:
                # Define data range
                yf.pdr_override()
                self.daily_data = pdr.get_data_yahoo(self.q, START, END)
                flag = True
            except:
                flag = False
                counter += 1
                if counter < 6:
                    continue
                else:
                    raise Exception("Yahoo finance is down, please try again later. ")

        return self.daily_data

    def technical_indicators_df(self):
        o = self.daily_data['Open'].values
        c = self.daily_data['Close'].values
        h = self.daily_data['High'].values
        l = self.daily_data['Low'].values
        v = self.daily_data['Volume'].astype(float).values
        # define the technical analysis matrix

        ta = pd.DataFrame()
        ta['MA5'] = tb.MA(c, timeperiod=5)
        ta['MA10'] = tb.MA(c, timeperiod=10)
        ta['MA20'] = tb.MA(c, timeperiod=20)
        ta['MA60'] = tb.MA(c, timeperiod=60)
        ta['MA120'] = tb.MA(c, timeperiod=120)
        ta['MA5'] = tb.MA(v, timeperiod=5)
        ta['MA10'] = tb.MA(v, timeperiod=10)
        ta['MA20'] = tb.MA(v, timeperiod=20)
        ta['ADX'] = tb.ADX(h, l, c, timeperiod=14)
        ta['ADXR'] = tb.ADXR(h, l, c, timeperiod=14)
        ta['MACD'] = tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0]
        ta['RSI'] = tb.RSI(c, timeperiod=14)
        ta['BBANDS_U'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0]
        ta['BBANDS_M'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1]
        ta['BBANDS_L'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2]
        ta['AD'] = tb.AD(h, l, c, v)
        ta['ATR'] = tb.ATR(h, l, c, timeperiod=14)
        ta['HT_DC'] = tb.HT_DCPERIOD(c)

        self.ta = ta

    def label(self, df, seq_length):
        return (df['Returns'] > 0).astype(int)

    def preprocessing(self):

        self.daily_data['Returns'] = pd.Series((self.daily_data['Close'] / self.daily_data['Close'].shift(1) - 1) * 100,
                                               index=self.daily_data.index)
        seq_length = 3
        self.daily_data['Volume'] = self.daily_data['Volume'].astype(float)
        self.X = self.daily_data[['Open', 'Close', 'High', 'Low', 'Volume']]
        self.y = self.label(self.daily_data, seq_length)
        X_shift = [self.X]
        for i in range(1, seq_length):
            X_shift.append(self.daily_data[['Open', 'Close', 'High', 'Low', 'Volume']].shift(i))
        ohlc = pd.concat(X_shift, axis=1)
        ohlc.columns = sum([[c + 'T-{}'.format(i) for c in ['Open', 'Close', 'High', 'Low', 'Volume']] \
                            for i in range(seq_length)], [])
        self.ta.index = ohlc.index
        self.X = pd.concat([ohlc, self.ta], axis=1)
        self.Xy = pd.concat([self.X, self.y], axis=1)

        fs = FeatureSelector(data=self.X, labels=self.y)
        fs.identify_all(selection_params={'missing_threshold': 0.6,
                                          'correlation_threshold': 0.9,
                                          'task': 'regression',
                                          'eval_metric': 'auc',
                                          'cumulative_importance': 0.99})
        self.X_fs = fs.remove(methods='all', keep_one_hot=True)
        self.Xy_fs = pd.concat([self.X_fs, self.y], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(self.X_fs, self.y, test_size=0.2, random_state=42)

        return X_train, y_train, X_test, y_test


class Display:

    def __init__(self, Xy, Xy_fs):
        self.Xy = Xy
        self.Xy_fs = Xy_fs

    def features_histograms(self):

        self.Xy.hist(bins=50, figsize=(20, 15), color='darkgreen')
        plt.savefig('features_histograms.png', bbox_inches='tight')
        plt.show()

    def plot_corr_heatmap(self):
        f, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(self.Xy.iloc[:, 0:-1].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
        plt.savefig('plot_corr_heatmap.png', bbox_inches='tight')
        plt.show()
    def plot_corr_heatmap_fs(self):

        f, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(self.Xy_fs.iloc[:, 0:-1].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
        plt.savefig('plot_corr_heatmap_fs.png', bbox_inches='tight')
        plt.show()

class XGB_training:
    def __init__(self, Xtrain, ytrain, Xtest, ytest):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest
        self._metric = ['error', 'logloss', 'auc']
        self.training()

    def calc_metrics(self, model):
        """
        This function fits model and returns the RMSE for in-sample error and out-of-sample error
        """
        train_error, train_score = self.calc_train_error(model)
        validation_error, validation_score = self.calc_validation_error(model)

        print("\n")
        print("Train MSE: ", round(train_error,4))
        print("Train Score: ", round(train_score,4))
        print("Test MSE", round(validation_error,4))
        print("Test Score", round(validation_score,4))

        return train_error, validation_error, train_score, validation_score

    def calc_train_error(self, model):
        """
        This function returns in-sample error for already fit model.
        """
        predictions = model.predict(self.Xtrain)
        mse = mean_squared_error(self.ytrain, predictions)
        score = accuracy_score(self.ytrain, predictions)
        return mse, score

    def calc_validation_error(self, model):
        """
        This function returns out-of-sample error for already fit model.
        """
        predictions = model.predict(self.Xtest)
        mse = mean_squared_error(self.ytest, predictions)
        score = accuracy_score(self.ytest, predictions)
        return mse, score

    def training(self):
        """
        Training is done at each max_depth loop.
        XGBoost's cv is used to find the optimum number of tree (estimators) at each depth, up to 1000 trees.
        Once traning result doesn't improve for 50 epochs, training will stop. The tree number used in the last epoch
        will be used to fit the train and test set again. Metrics will then be measured again this XGB model.
        """

        max_depth = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        best_depth = 0
        best_estimator = 0
        max_score = 0
        for md in max_depth:
            model = XGBClassifier(learning_rate=0.3, n_estimators=1000, max_depth=md, min_child_weight=1,
                                  gamma=1, subsample=1, colsample_bytree=0.1, reg_lambda=0, reg_alpha=1,
                                  random_state=42)
            xgb_param = model.get_xgb_params()
            xgtrain = xgboost.DMatrix(self.Xtrain.values, label=self.ytrain.values)

            cvresult = xgboost.cv(xgb_param, xgtrain, num_boost_round=1000, early_stopping_rounds=50,
                                  nfold=8, metrics='auc', stratified=True, shuffle=True, seed=42,
                                  verbose_eval=False)
            print("There are {} trees in the XGB model. CV-mean: {:.4f}, CV-std: {:.4f}.".format(
                cvresult.shape[0], cvresult.iloc[cvresult.shape[0] - 1, 0],
                cvresult.iloc[cvresult.shape[0] - 1, 1]))
            n = cvresult.shape[0]
            model.set_params(n_estimators=n)
            model.fit(self.Xtrain,
                      self.ytrain,
                      eval_metric=self._metric,
                      eval_set=[(self.Xtrain, self.ytrain), (self.Xtest, self.ytest)],
                      verbose=False)
            y_pred = model.predict(self.Xtest)
            score = accuracy_score(self.ytest, y_pred)
            mse = mean_squared_error(self.ytest, y_pred)

            if score > max_score:
                max_score = score
                min_mse = mse
                best_depth = md
                best_estimator = n
                self.best_xgb = model
            print("Accuracy score: " + str(round(score, 4)) + " at depth: " + str(md) + " and estimator " + str(n))
            print("Mean square error: " + str(round(mse, 4)) + " at depth: " + str(md) + " and estimator " + str(n))
        print("Best score: " + str(round(max_score, 4)) + " Best MSE: " + str(round(min_mse, 4)) + " at depth: " + str(
            best_depth) + " and estimator of " + str(best_estimator))


    def predict(self):
        """
        Predicts the labels for the original test set
        """
        print("\n")
        print("Best XGB model:")
        pprint(self.best_xgb.get_xgb_params())
        self.calc_metrics(self.best_xgb)

        # plot boosting results
        results = self.best_xgb.evals_result()
        epochs = len(results['validation_0'][self._metric[0]])
        x_axis = range(0, epochs)
        plt.style.use('ggplot')
        plt.rcParams['font.size'] = 8
        i = 0
        plt.figure(figsize=(20, 15))
        for m in self._metric:
            ax = plt.subplot2grid((len(self._metric), 2), (i, 0))
            i += 1
            ax.plot(x_axis, results['validation_0'][m], label='Train')
            ax.plot(x_axis, results['validation_1'][m], label='Test')
            ax.legend()
            ax.set_ylabel(m)
            plt.savefig('training.png', bbox_inches='tight')
        plt.show()

        # plot feature importances
        ax = xgboost.plot_importance(self.best_xgb.get_booster())
        fig = ax.figure
        fig.set_size_inches(14, 8)
        plt.savefig('plot_importance.png', bbox_inches='tight')
        plt.show()

        # plot tree
        ax = xgboost.plot_tree(self.best_xgb.get_booster(), num_trees=4)
        fig = ax.figure
        fig.set_size_inches(8, 8)
        plt.savefig('tree.png', bbox_inches='tight')
        plt.show()


class FeatureSelector():
    """
    Courtesy of William Koehrsen from Feature Labs
    Class for performing feature selection for machine learning or data preprocessing.

    Implements five different methods to identify features for removal

        1. Find columns with a missing percentage greater than a specified threshold
        2. Find columns with a single unique value
        3. Find collinear variables with a correlation greater than a specified correlation coefficient
        4. Find features with 0.0 feature importance from a gradient boosting machine (gbm)
        5. Find low importance features that do not contribute to a specified cumulative feature importance from the gbm

    Parameters
    --------
        data : dataframe
            A dataset with observations in the rows and features in the columns

        labels : array or series, default = None
            Array of labels for training the machine learning model to find feature importances. These can be either binary labels
            (if task is 'classification') or continuous targets (if task is 'regression').
            If no labels are provided, then the feature importance based methods are not available.

    Attributes
    --------

    ops : dict
        Dictionary of operations run and features identified for removal

    missing_stats : dataframe
        The fraction of missing values for all features

    record_missing : dataframe
        The fraction of missing values for features with missing fraction above threshold

    unique_stats : dataframe
        Number of unique values for all features

    record_single_unique : dataframe
        Records the features that have a single unique value

    corr_matrix : dataframe
        All correlations between all features in the data

    record_collinear : dataframe
        Records the pairs of collinear variables with a correlation coefficient above the threshold

    feature_importances : dataframe
        All feature importances from the gradient boosting machine

    record_zero_importance : dataframe
        Records the zero importance features in the data according to the gbm

    record_low_importance : dataframe
        Records the lowest importance features not needed to reach the threshold of cumulative importance according to the gbm


    Notes
    --------

        - All 5 operations can be run with the `identify_all` method.
        - If using feature importances, one-hot encoding is used for categorical variables which creates new columns

    """

    def __init__(self, data, labels=None):

        # Dataset and optional training labels
        self.data = data
        self.labels = labels

        if labels is None:
            print('No labels provided. Feature importance based methods are not available.')

        self.base_features = list(data.columns)
        self.one_hot_features = None

        # Dataframes recording information about features to remove
        self.record_missing = None
        self.record_single_unique = None
        self.record_collinear = None
        self.record_zero_importance = None
        self.record_low_importance = None

        self.missing_stats = None
        self.unique_stats = None
        self.corr_matrix = None
        self.feature_importances = None

        # Dictionary to hold removal operations
        self.ops = {}

        self.one_hot_correlated = False

    def identify_missing(self, missing_threshold):
        """Find the features with a fraction of missing values above `missing_threshold`"""

        self.missing_threshold = missing_threshold

        # Calculate the fraction of missing in each column
        missing_series = self.data.isnull().sum() / self.data.shape[0]
        self.missing_stats = pd.DataFrame(missing_series).rename(columns={'index': 'feature', 0: 'missing_fraction'})

        # Sort with highest number of missing values on top
        self.missing_stats = self.missing_stats.sort_values('missing_fraction', ascending=False)

        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(columns=
            {'index': 'feature', 0: 'missing_fraction'})

        to_drop = list(record_missing['feature'])

        self.record_missing = record_missing
        self.ops['missing'] = to_drop

        print('%d features with greater than %0.2f missing values.\n' % (
        len(self.ops['missing']), self.missing_threshold))

    def identify_single_unique(self):
        """Finds features with only a single unique value. NaNs do not count as a unique value. """

        # Calculate the unique counts in each column
        unique_counts = self.data.nunique()
        self.unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
        self.unique_stats = self.unique_stats.sort_values('nunique', ascending=True)

        # Find the columns with only one unique count
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
            columns={'index': 'feature',
                     0: 'nunique'})

        to_drop = list(record_single_unique['feature'])

        self.record_single_unique = record_single_unique
        self.ops['single_unique'] = to_drop

        print('%d features with a single unique value.\n' % len(self.ops['single_unique']))

    def identify_collinear(self, correlation_threshold, one_hot=False):
        """
        Finds collinear features based on the correlation coefficient between features.
        For each pair of features with a correlation coefficient greather than `correlation_threshold`,
        only one of the pair is identified for removal.

        Using code adapted from: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

        Parameters
        --------

        correlation_threshold : float between 0 and 1
            Value of the Pearson correlation cofficient for identifying correlation features

        one_hot : boolean, default = False
            Whether to one-hot encode the features before calculating the correlation coefficients

        """

        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot

        # Calculate the correlations between every column
        if one_hot:

            # One hot encoding
            features = pd.get_dummies(self.data)
            self.one_hot_features = [column for column in features.columns if column not in self.base_features]

            # Add one hot encoded data to original data
            self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)

            corr_matrix = pd.get_dummies(features).corr()

        else:
            corr_matrix = self.data.corr()

        self.corr_matrix = corr_matrix

        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:
            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})

            # Add to dataframe
            record_collinear = record_collinear.append(temp_df, ignore_index=True)

        self.record_collinear = record_collinear
        self.ops['collinear'] = to_drop

        print('%d features with a correlation magnitude greater than %0.2f.\n' % (
        len(self.ops['collinear']), self.correlation_threshold))

    def identify_zero_importance(self, task, eval_metric=None,
                                 n_iterations=10, early_stopping=True):
        """

        Identify the features with zero importance according to a gradient boosting machine.
        The gbm can be trained with early stopping using a validation set to prevent overfitting.
        The feature importances are averaged over `n_iterations` to reduce variance.

        Uses the LightGBM implementation (http://lightgbm.readthedocs.io/en/latest/index.html)

        Parameters
        --------

        eval_metric : string
            Evaluation metric to use for the gradient boosting machine for early stopping. Must be
            provided if `early_stopping` is True

        task : string
            The machine learning task, either 'classification' or 'regression'

        n_iterations : int, default = 10
            Number of iterations to train the gradient boosting machine

        early_stopping : boolean, default = True
            Whether or not to use early stopping with a validation set when training


        Notes
        --------

        - Features are one-hot encoded to handle the categorical variables before training.
        - The gbm is not optimized for any particular task and might need some hyperparameter tuning
        - Feature importances, including zero importance features, can change across runs

        """

        if early_stopping and eval_metric is None:
            raise ValueError("""eval metric must be provided with early stopping. Examples include "auc" for classification or
                             "l2" for regression.""")

        if self.labels is None:
            raise ValueError("No training labels provided.")

        # One hot encoding
        features = pd.get_dummies(self.data)
        self.one_hot_features = [column for column in features.columns if column not in self.base_features]

        # Add one hot encoded data to original data
        self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)

        # Extract feature names
        feature_names = list(features.columns)

        # Convert to np array
        features = np.array(features)
        labels = np.array(self.labels).reshape((-1,))

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))

        print('Training Gradient Boosting Model\n')

        # Iterate through each fold
        for _ in range(n_iterations):

            if task == 'classification':
                model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=0)

            elif task == 'regression':
                model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=0)

            else:
                raise ValueError('Task must be either "classification" or "regression"')

            # If training using early stopping need a validation set
            if early_stopping:

                train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels,
                                                                                              test_size=0.15)
                # Train the model with early stopping
                model.fit(train_features, train_labels, eval_metric=eval_metric,
                          eval_set=[(valid_features, valid_labels)],
                          early_stopping_rounds=100, verbose=0)

                # Clean up memory
                gc.enable()
                del train_features, train_labels, valid_features, valid_labels
                gc.collect()

            else:
                model.fit(features, labels)

            # Record the feature importances
            feature_importance_values += model.feature_importances_ / n_iterations

        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Sort features according to importance
        feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

        # Normalize the feature importances to add up to one
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances[
            'importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

        # Extract the features with zero importance
        record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]

        to_drop = list(record_zero_importance['feature'])

        self.feature_importances = feature_importances
        self.record_zero_importance = record_zero_importance
        self.ops['zero_importance'] = to_drop

        print('\n%d features with zero importance after one-hot encoding.\n' % len(self.ops['zero_importance']))

    def identify_low_importance(self, cumulative_importance):
        """
        Finds the lowest importance features not needed to account for `cumulative_importance` fraction
        of the total feature importance from the gradient boosting machine. As an example, if cumulative
        importance is set to 0.95, this will retain only the most important features needed to
        reach 95% of the total feature importance. The identified features are those not needed.

        Parameters
        --------
        cumulative_importance : float between 0 and 1
            The fraction of cumulative importance to account for

        """

        self.cumulative_importance = cumulative_importance

        # The feature importances need to be calculated before running
        if self.feature_importances is None:
            raise NotImplementedError("""Feature importances have not yet been determined. 
                                         Call the `identify_zero_importance` method first.""")

        # Make sure most important features are on top
        self.feature_importances = self.feature_importances.sort_values('cumulative_importance')

        # Identify the features not needed to reach the cumulative_importance
        record_low_importance = self.feature_importances[
            self.feature_importances['cumulative_importance'] > cumulative_importance]

        to_drop = list(record_low_importance['feature'])

        self.record_low_importance = record_low_importance
        self.ops['low_importance'] = to_drop

        print('%d features required for cumulative importance of %0.2f after one hot encoding.' % (
        len(self.feature_importances) -
        len(self.record_low_importance), self.cumulative_importance))
        print('%d features do not contribute to cumulative importance of %0.2f.\n' % (len(self.ops['low_importance']),
                                                                                      self.cumulative_importance))

    def identify_all(self, selection_params):
        """
        Use all five of the methods to identify features to remove.

        Parameters
        --------

        selection_params : dict
           Parameters to use in the five feature selection methhods.
           Params must contain the keys ['missing_threshold', 'correlation_threshold', 'eval_metric', 'task', 'cumulative_importance']

        """

        # Check for all required parameters
        for param in ['missing_threshold', 'correlation_threshold', 'eval_metric', 'task', 'cumulative_importance']:
            if param not in selection_params.keys():
                raise ValueError('%s is a required parameter for this method.' % param)

        # Implement each of the five methods
        self.identify_missing(selection_params['missing_threshold'])
        self.identify_single_unique()
        self.identify_collinear(selection_params['correlation_threshold'])
        self.identify_zero_importance(task=selection_params['task'], eval_metric=selection_params['eval_metric'])
        self.identify_low_importance(selection_params['cumulative_importance'])

        # Find the number of features identified to drop
        self.all_identified = set(list(chain(*list(self.ops.values()))))
        self.n_identified = len(self.all_identified)

        print('%d total features out of %d identified for removal after one-hot encoding.\n' % (self.n_identified,
                                                                                                self.data_all.shape[1]))

    def check_removal(self, keep_one_hot=True):

        """Check the identified features before removal. Returns a list of the unique features identified."""

        self.all_identified = set(list(chain(*list(self.ops.values()))))
        print('Total of %d features identified for removal' % len(self.all_identified))

        if not keep_one_hot:
            if self.one_hot_features is None:
                print('Data has not been one-hot encoded')
            else:
                one_hot_to_remove = [x for x in self.one_hot_features if x not in self.all_identified]
                print('%d additional one-hot features can be removed' % len(one_hot_to_remove))

        return list(self.all_identified)

    def remove(self, methods, keep_one_hot=True):
        """
        Remove the features from the data according to the specified methods.

        Parameters
        --------
            methods : 'all' or list of methods
                If methods == 'all', any methods that have identified features will be used
                Otherwise, only the specified methods will be used.
                Can be one of ['missing', 'single_unique', 'collinear', 'zero_importance', 'low_importance']
            keep_one_hot : boolean, default = True
                Whether or not to keep one-hot encoded features

        Return
        --------
            data : dataframe
                Dataframe with identified features removed


        Notes
        --------
            - If feature importances are used, the one-hot encoded columns will be added to the data (and then may be removed)
            - Check the features that will be removed before transforming data!

        """

        features_to_drop = []

        if methods == 'all':

            # Need to use one-hot encoded data as well
            data = self.data_all

            print('{} methods have been run\n'.format(list(self.ops.keys())))

            # Find the unique features to drop
            features_to_drop = set(list(chain(*list(self.ops.values()))))

        else:
            # Need to use one-hot encoded data as well
            if 'zero_importance' in methods or 'low_importance' in methods or self.one_hot_correlated:
                data = self.data_all

            else:
                data = self.data

            # Iterate through the specified methods
            for method in methods:

                # Check to make sure the method has been run
                if method not in self.ops.keys():
                    raise NotImplementedError('%s method has not been run' % method)

                # Append the features identified for removal
                else:
                    features_to_drop.append(self.ops[method])

            # Find the unique features to drop
            features_to_drop = set(list(chain(*features_to_drop)))

        features_to_drop = list(features_to_drop)

        if not keep_one_hot:

            if self.one_hot_features is None:
                print('Data has not been one-hot encoded')
            else:

                features_to_drop = list(set(features_to_drop) | set(self.one_hot_features))

        # Remove the features and return the data
        data = data.drop(columns=features_to_drop)
        self.removed_features = features_to_drop

        if not keep_one_hot:
            print('Removed %d features including one-hot features.' % len(features_to_drop))
        else:
            print('Removed %d features.' % len(features_to_drop))

        return data

    def plot_missing(self):
        """Histogram of missing fraction in each feature"""
        if self.record_missing is None:
            raise NotImplementedError("Missing values have not been calculated. Run `identify_missing`")

        self.reset_plot()

        # Histogram of missing values
        plt.style.use('seaborn-white')
        plt.figure(figsize=(7, 5))
        plt.hist(self.missing_stats['missing_fraction'], bins=np.linspace(0, 1, 11), edgecolor='k', color='red',
                 linewidth=1.5)
        plt.xticks(np.linspace(0, 1, 11));
        plt.xlabel('Missing Fraction', size=14);
        plt.ylabel('Count of Features', size=14);
        plt.title("Fraction of Missing Values Histogram", size=16);

    def plot_unique(self):
        """Histogram of number of unique values in each feature"""
        if self.record_single_unique is None:
            raise NotImplementedError('Unique values have not been calculated. Run `identify_single_unique`')

        self.reset_plot()

        # Histogram of number of unique values
        self.unique_stats.plot.hist(edgecolor='k', figsize=(7, 5))
        plt.ylabel('Frequency', size=14);
        plt.xlabel('Unique Values', size=14);
        plt.title('Number of Unique Values Histogram', size=16);

    def plot_collinear(self, plot_all=False):
        """
        Heatmap of the correlation values. If plot_all = True plots all the correlations otherwise
        plots only those features that have a correlation above the threshold

        Notes
        --------
            - Not all of the plotted correlations are above the threshold because this plots
            all the variables that have been idenfitied as having even one correlation above the threshold
            - The features on the x-axis are those that will be removed. The features on the y-axis
            are the correlated features with those on the x-axis

        Code adapted from https://seaborn.pydata.org/examples/many_pairwise_correlations.html
        """

        if self.record_collinear is None:
            raise NotImplementedError('Collinear features have not been idenfitied. Run `identify_collinear`.')

        if plot_all:
            corr_matrix_plot = self.corr_matrix
            title = 'All Correlations'

        else:
            # Identify the correlations that were above the threshold
            # columns (x-axis) are features to drop and rows (y_axis) are correlated pairs
            corr_matrix_plot = self.corr_matrix.loc[list(set(self.record_collinear['corr_feature'])),
                                                    list(set(self.record_collinear['drop_feature']))]

            title = "Correlations Above Threshold"

        f, ax = plt.subplots(figsize=(10, 8))

        # Diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with a color bar
        sns.heatmap(corr_matrix_plot, cmap=cmap, center=0,
                    linewidths=.25, cbar_kws={"shrink": 0.6})

        # Set the ylabels
        ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
        ax.set_yticklabels(list(corr_matrix_plot.index), size=int(160 / corr_matrix_plot.shape[0]));

        # Set the xlabels
        ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
        ax.set_xticklabels(list(corr_matrix_plot.columns), size=int(160 / corr_matrix_plot.shape[1]));
        plt.title(title, size=14)

    def plot_feature_importances(self, plot_n=15, threshold=None):
        """
        Plots `plot_n` most important features and the cumulative importance of features.
        If `threshold` is provided, prints the number of features needed to reach `threshold` cumulative importance.

        Parameters
        --------

        plot_n : int, default = 15
            Number of most important features to plot. Defaults to 15 or the maximum number of features whichever is smaller

        threshold : float, between 0 and 1 default = None
            Threshold for printing information about cumulative importances

        """

        if self.record_zero_importance is None:
            raise NotImplementedError('Feature importances have not been determined. Run `idenfity_zero_importance`')

        # Need to adjust number of features if greater than the features in the data
        if plot_n > self.feature_importances.shape[0]:
            plot_n = self.feature_importances.shape[0] - 1

        self.reset_plot()

        # Make a horizontal bar chart of feature importances
        plt.figure(figsize=(10, 6))
        ax = plt.subplot()

        # Need to reverse the index to plot most important on top
        # There might be a more efficient method to accomplish this
        ax.barh(list(reversed(list(self.feature_importances.index[:plot_n]))),
                self.feature_importances['normalized_importance'][:plot_n],
                align='center', edgecolor='k')

        # Set the yticks and labels
        ax.set_yticks(list(reversed(list(self.feature_importances.index[:plot_n]))))
        ax.set_yticklabels(self.feature_importances['feature'][:plot_n], size=12)

        # Plot labeling
        plt.xlabel('Normalized Importance', size=16);
        plt.title('Feature Importances', size=18)
        plt.show()

        # Cumulative importance plot
        plt.figure(figsize=(6, 4))
        plt.plot(list(range(1, len(self.feature_importances) + 1)), self.feature_importances['cumulative_importance'],
                 'r-')
        plt.xlabel('Number of Features', size=14);
        plt.ylabel('Cumulative Importance', size=14);
        plt.title('Cumulative Feature Importance', size=16);

        if threshold:
            # Index of minimum number of features needed for cumulative importance threshold
            # np.where returns the index so need to add 1 to have correct number
            importance_index = np.min(np.where(self.feature_importances['cumulative_importance'] > threshold))
            plt.vlines(x=importance_index + 1, ymin=0, ymax=1, linestyles='--', colors='blue')
            plt.show();

            print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))

    def reset_plot(self):
        plt.rcParams = plt.rcParamsDefault

# ----------------------------- MAIN PROGRAM ---------------------------------
def main():
    """
    The main program

    """
    print("\n")
    print("##################### Gradient Boosting Classification by XGBoost on stock data ##########################")
    print("\n")
    # Set the print canvas right
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    pd.set_option('display.max_columns', 14)
    pd.set_option('display.width', 1600)

    print("*********************************************  Data Preprocessing ***************************************")
    print("\n")

    symbol = UserInput.get_symbol()
    print("Downloading stock data ...")
    stock_data = Data(symbol)
    print("\n")
    print("Preprocessing and selecting features ...")
    print("\n")
    X_train, y_train, X_test, y_test = stock_data.preprocessing()
    X, y, Xy, Xy_fs = stock_data.X, stock_data.y, stock_data.Xy, stock_data.Xy_fs
    print("\n")
    print("Original features >")
    print(Xy.info())
    print("\n")
    print("Selected features >")
    print(Xy_fs.info())
    print("\n")

    plot = Display(Xy, Xy_fs)
    plot.features_histograms()
    plot.plot_corr_heatmap()
    plot.plot_corr_heatmap_fs()

    print("******************************************  Model training and tuning ************************************")
    print("\n")

    # Training Decision Tree & Random Forest model to get the best model and its hyperparameters:
    xgb_clf = XGB_training(X_train, y_train, X_test, y_test)
    xgb_clf.predict()

    print("\n")
    print(
        "All plots are saved with relevant filenames in the same folder as this program, feel free to review after this program ends.")
    print("\n")
    print(
        "#########################################   END OF PROGRAM   ##############################################")

if __name__ == '__main__':
    main()