import numpy as np
import pandas as pd
import pickle
from copy import deepcopy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import xgboost as xgb

from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

class Dataset(pd.DataFrame):

    def __init__(self, name, type, df):
        super().__init__(df)

        self.name = name
        self.type = type
        self.checkNans()


    def checkNans(self):
        self.nans_idx = {}
        self.filled_idx = {}
        self.all_filled_idx = []

        # finds the missing and filled features in the Dataset
        for feature in self.columns:
            self.nans_idx.update({feature : list(
                self.index[self[feature].isnull()])})
            self.filled_idx.update({feature : list(
                self.index[self[feature].notnull()])})

        # finds the indeces that have all features filled
        features = self.getFeatures()

        idxs = None
        for feature in features:
            if idxs:
                idxs &= set(self.filled_idx[feature])
            else:
                idxs = set(self.filled_idx[feature])
        self.all_filled_idx = list(idxs)


    def getImputationTestset(self, missing_values=1, features=None):
        '''
        Creates another Dataset that can be used for imputation testing
        1. only the samples that have all features are considered
        2. for each sample a random number of features is selected (max massing_values)
        3. the selected features are set to None

        Parameters
        ----------
        missing_values : int
            specifies how many features maximum are set to None for each sample
            the actual number of features set to None follows a uniform distribution

        features : list (optional)
            the features that should be considered

        '''

        if features == None:
            features = self.getFeatures()

        self.imp_test_set = Dataset(self.name, 'imp_test_set',
            self.loc[self.all_filled_idx, :].copy(deep=True))

        for idx in self.imp_test_set.index:
            missing_vals = np.random.randint(1, missing_values+1, 1)[0] if missing_values > 1 else 1
            missing_feats = np.random.randint(0, len(features), missing_vals) if len(features) > 1 else [0]
            for missing_feat in missing_feats:
                self.imp_test_set.loc[idx, features[missing_feat]] = None

        self.imp_test_set.checkNans()
        return self.imp_test_set


    def getYShuffledSet(self, yfeatures, swaps):
        '''
        Creates another Dataset that can be used for outlier testing
        1. only the samples that have all features are considered
        2. random "swaps" samples' "yfeatures" are swapped with other random
           "swaps" samples' "yfeatures"

        Parameters
        ----------
        yfeatures : list
            list of features that constituing the "dependent" variables

        swaps : int
            amount of swaps to be performed


        Returns
        -------
        yshuffled_idx : list
        List of tuples that were swapped (e.g. [(1,98), (2, 99), (98, 1), (99, 2)])
        yshuffled : Dataset
        The y shuffled dataset

        '''

        self.yshuffled_set = Dataset(self.name, 'yshuffled',
            self.loc[self.all_filled_idx, :].copy(deep=True))

        idx = np.random.choice(self.yshuffled_set.index, 20, replace=False)
        idx_1 = idx[0:10]
        idx_2 = idx[10:]

        swap = self.yshuffled_set.loc[idx_1, yfeatures].copy().values
        self.yshuffled_set.loc[idx_1, yfeatures] = self.yshuffled_set.loc[idx_2, yfeatures].copy().values
        self.yshuffled_set.loc[idx_2, yfeatures] = swap
        return list(zip(idx_1, idx_2)) + list(zip(idx_2, idx_1)), self.yshuffled_set


    def getFeatures(self):
        return list(self.columns)

    def getFeaturesWithout(self, features):
        feats = []
        for f in self.columns:
            if f not in features:
                feats.append(f)
        return feats


    def getMissing(self, feature, selection=None, positive=True):
        '''
        Returns the samples which's feature was empty

        Parameters
        ----------
        feature : string
            the feature that had been missing

        selection : list or single feature name
            the features that should be returned

        positve : boolean
            whether or not the selection is all or all except
        '''

        if selection:
            if positive:
                return self.loc[self.nans_idx[feature], selection]
            else:
                return self.loc[self.nans_idx[feature], self.getFeaturesWithout(selection)]
        else:
            return self.loc[self.nans_idx[feature], :]


    def getFilled(self, feature, selection=None, positive=True):
        '''
        Returns the samples which's feature was filled

        Parameters
        ----------
        feature : string
            the feature that had been filled

        selection : list or single feature name
            the features that should be returned

        positve : boolean
            whether or not the selection is all or all except
        '''

        if selection:
            if positive:
                return self.loc[self.filled_idx[feature], selection]
            else:
                return self.loc[self.filled_idx[feature], self.getFeaturesWithout(selection)]
        else:
            return self.loc[self.filled_idx[feature], :]


    def fillMissingValues(self, funktion, features = None):
        ''' Fills the missing values in the dataset with the value specified by method

        Parameters
        ----------
        funktion : function
        a function taking in a pandas Series and returning a single values

        features : list (optimal)
        a list with all the features that should be filled
        if omitted all are filled
        '''

        if features == None:
            features = self.getFeatures()
        for feature in features:
            if len(self.filled_idx[feature]) > 0:
                self.loc[self.nans_idx[feature], feature] = \
                    funktion(self.loc[self.filled_idx[feature], feature])


    def restoreMissingValues(self, features = None):
        ''' Restores the missing values in the dataset with NaNs

        Parameters
        ----------

        features : list (optimal)
        a list with all the features that should be restored
        if omitted all are restored
        '''

        if features == None:
            features = self.getFeatures()
        for feature in features:
            if len(self.filled_idx[feature]) > 0:
                self.loc[self.nans_idx[feature], feature] = None


    @staticmethod
    def plotDensities(datasets, features=None, x_range=None, y_range=None):
        if features == None:
            features = datasets[0].getFeatures()

        plot_top = 0.8
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(5*len(features), 4*len(datasets)))
        fig.text(x=0.5, y=(1.0 - (1.0 - plot_top)/2.0), s="Densities of features",
            va='center', ha='center', fontsize=32)

        gspec = gs.GridSpec(nrows=len(datasets), ncols=len(features),
            left=0.1, right=0.95, top=plot_top, bottom=0.05)
        title_y = np.linspace(plot_top, 0.0, len(datasets) + 1)[1:]
        title_y = title_y + plot_top / (2 * len(datasets))

        for l, ds in enumerate(datasets):
            fig.text(x=0.025, y=title_y[l], s=ds.type,
                va='center', ha='center', rotation='vertical', fontsize=32)
            for k, feature in enumerate(features):
                # ax = fig.add_subplot(len(datasets), len(features), l*len(features) + k + 1)
                ax = plt.subplot(gspec[l, k])
                density = stats.kde.gaussian_kde(
                    ds.loc[ds.filled_idx[feature], feature])
                x = np.arange(np.min(ds[feature]),
                    np.max(ds[feature]), .1)
                ax.set_title(feature)
                if x_range is not None:
                    ax.set_autoscalex_on(False)
                    ax.set_xlim(x_range)
                if y_range is not None:
                    ax.set_autoscaley_on(False)
                    ax.set_ylim(y_range)
                ax.plot(x, density(x))

        fig.set_facecolor('w')
        return fig


class BasicAnalyzer:
    '''
    The class BasicAnalyzer offers methods to explore and anlyse datasets
    '''

    train = None    # the training dataset as pandas dataframe
    valid = None    # the validation dataset as pandas dataframe
    test = None     # the test dataset as pandas dataframe

    features_max = None     # features in all: training, valid AND test set
    features_min = None     # features in one: training, valid OR test set

    params_std = \
"""
# Standard Parameters
self.params = [
    # {
    # 'bagging__base_estimator': [DecisionTreeRegressor()],
    # 'bagging__base_estimator__splitter': ['random'],
    # 'bagging__base_estimator__max_depth': [1, 3, 5, 10, 20, None]
    # },
    # {
    # 'bagging__base_estimator': [KNeighborsRegressor()],
    # 'bagging__base_estimator__n_neighbors': [2, 3, 5, 10, 20]
    # },
    # {
    # 'bagging__base_estimator': [LinearRegression()],
    # },
    # {
    # 'bagging__base_estimator': [SVR()],
    # 'bagging__base_estimator__kernel': ['rbf'],
    # 'bagging__base_estimator__C': [10, 100]
    # },
    {
    'bagging__base_estimator': [xgb.sklearn.XGBRegressor()],
    'bagging__base_estimator__max_depth': [1, 3, 5, 10, 20],
    'bagging__base_estimator__objective': ['reg:squarederror'],
    }
]

for param in self.params:
    param.update({
    'features': [None, StandardScaler()],
    'bagging__n_estimators':  [100],
    'bagging__max_samples': [0.1],
    'bagging__bootstrap': [True],
    'bagging__n_jobs': [-1],
    })

# This command will be executed to perform the grid search
# self.pipeline = Pipeline([
#     ('features', None),
#     ('bagging', BaggingRegressor())
# ])
# gcv = GridSearchCV(self.pipeline, self.params, cv=5)
"""

    iterations = 2

    def __init__(self, name, train=None, valid=None):
        self.name = name
        self.gcvs = {
            "models": {}, # the grid searches
            "params": {},  # the paramters used to perform the grid searches
            "params_std": BasicAnalyzer.params_std
        }
        self.params_code = {}

        if train is not None:
            BasicAnalyzer.setTrainDS(train)
        if valid is not None:
            BasicAnalyzer.setValidDS(valid)


    def close(self):
        pass


    @staticmethod
    def setTrainDS(train):
        BasicAnalyzer.train = train
        BasicAnalyzer.updataFeatures()


    @staticmethod
    def setValidDS(valid):
        BasicAnalyzer.valid = valid
        BasicAnalyzer.updataFeatures()


    @staticmethod
    def updataFeatures():

        # finds the features that are available in any set
        features = []
        features += BasicAnalyzer.train.getFeatures() if BasicAnalyzer.train is not None else []
        features += BasicAnalyzer.valid.getFeatures() if BasicAnalyzer.valid is not None else []
        BasicAnalyzer.features_max = list(dict.fromkeys(features))

        # finds the features that are available in all set
        features = None
        for ds in [BasicAnalyzer.train, BasicAnalyzer.valid]:
            if ds is not None:
                if features:
                    features &= set(ds.getFeatures())
                else:
                    features = set(ds.getFeatures())
        BasicAnalyzer.features_min = list(features) if features is not None else []


    def saveGCV(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.gcvs, f)


    def loadGCV(self, filename=None):
        if filename is None:
            filename = "gcvs.pickle"
        file = open(filename,"rb")
        self.gcvs = pickle.load(file)


    def fitFeatureModels(self, features=None):
        print("BasicAnalyer.fitFeatureModels")
        if BasicAnalyzer.train is not None:
            if features == None:
                features = BasicAnalyzer.train.getFeatures()

            for feature in features:
                self.fitFeatureModel(feature)


    def fitFeatureModel(self, feature):
        ''' Searches for and fits the model  that best fits the predictiton of
        the selected feature

        Parameters
        ----------
        feature : string
        the feature that is being predicted
        '''

        print("Fitting Missing Value Model for {}".format(feature))

        if not feature in self.gcvs["params"]:
            self.gcvs["params"].update({feature: self.gcvs["params_std"]})
        exec(self.gcvs["params"][feature])

        # WARNING!!!!
        # THE FOLLOWING PIPELINE LOGIC IS EXPECTED BY OTHER PROGRAM PARTS!!!
        # WARNING!!!!
        self.pipeline = Pipeline([
            ('features', None),
            ('bagging', BaggingRegressor())
        ])

        gcv = GridSearchCV(self.pipeline, self.params, cv=5)
        gcv.fit(
            BasicAnalyzer.train.getFilled(feature, [feature], False),
            BasicAnalyzer.train.getFilled(feature, feature)
        )
        self.gcvs["models"].update({feature: gcv})


    def iterateMissingValuePredictions(self, ds, features=None, iterations=5):
        ''' Forward iterates the predictions of missing values

        Parameters
        ----------
        features : list (optimal)
        a list with all the features that should be predicted forward
        if omitted all are used
        '''

        if features == None:
            features = BasicAnalyzer.train.getFeatures()

        if iterations == None:
            iterations = BasicAnalyzer.iterations
        else:
            BasicAnalyzer.iterations = iterations

        ds.fillMissingValues(np.average, features)
        self.imputed_std = pd.DataFrame(np.array([None]*(ds.shape[0]*ds.shape[1])).reshape(ds.shape[0], ds.shape[1]))
        self.imputed_std.index = ds.index
        self.imputed_std.columns = ds.columns

        for k in range(iterations):
            for feature in features:
                print("Iteration: {} Feature: {}".format(k, feature))
                if k == (BasicAnalyzer.iterations - 1):
                    self.iterationMissingValuePredictions(ds, self.imputed_std, feature)
                else:
                    self.iterationMissingValuePredictions(ds, None, feature)

        cols = []
        for meaning in ["orig", "imputed", "std"]:
            for feature in list(ds.columns):
                cols += ["{}_{}".format(meaning, feature)]
        self.imputed = pd.concat([ds.copy(), ds.copy(), self.imputed_std], axis=1)

        self.imputed.columns = cols
        for feature in list(ds.columns):
            self.imputed.loc[ds.nans_idx[feature], "{}_{}".format("orig", feature)] = None
            self.imputed.loc[ds.filled_idx[feature], "{}_{}".format("imputed", feature)] = None


    def iterationMissingValuePredictions(self, ds, ds_std, feature):
        ''' Performs a single forward iteration step for the prediction of the
        missing values

        Parameters
        ----------
        feature : string
        the feature that is being predicted
        '''

        if len(ds.nans_idx[feature]) > 0:
            test = ds.getMissing(feature, [feature], False).values
            pred = self.gcvs["models"][feature].best_estimator_.predict(
                test
            )
            ds.loc[ds.nans_idx[feature], feature] = pred
            if ds_std is not None:
                preds = []
                for est in self.gcvs["models"][feature].best_estimator_['bagging'].estimators_:
                    test = ds.getMissing(feature, [feature], False).values
                    preds += [est.predict(test)]
                preds = np.array(preds)
                ds_std.loc[ds.nans_idx[feature], feature] = np.apply_along_axis(np.std, 0, preds)


    def findOutliers(self, test):
        ''' for each value in the test set, the method returns in a Dataset
        the percentage of estimators in the Bagging object that estimated a value
        further away from the bagged value than the value in the dataset.

        If there are few estimators that predict a more extreme value, than this
        would point to the value being an outlier.

        Parameters
        ----------
        test : Dataset
        the dataset that should be analyzed for potential outliers
        '''

        file = open("debug\\findOutliers_basic.csv", "w")

        test_pc = Dataset(test.name, "outlier_pc", test.loc[test.all_filled_idx, :].copy())
        test_std = Dataset(test.name, "outlier_pc", test.loc[test.all_filled_idx, :].copy())

        pred_bagging, preds = self.predictAllEstimators(test_pc)

        pred_bagging.to_csv("debug\\findOutliers_basic_pred_bagging.csv")
        file.write("{},{},{},{}".format("sample", "feature", "prediction", "more extreme"))
        pred_text = ""
        predn_text = ""
        for i, pred in enumerate(preds):
            pred.to_csv("debug\\findOutliers_basic_preds_{}.csv".format(i))
            pred_text += ",{}_{}".format("pred", i)
            predn_text += ",{}_{}".format("predn", i)

        file.write(",{}{}".format("actual value", pred_text))
        file.write(",{}{}".format("actual value n", predn_text))
        file.write("\n")

        features = BasicAnalyzer.train.getFeatures()
        for feature in features:
            for sample in test.all_filled_idx:
                prediction = pred_bagging.loc[sample, feature]
                predictions = [x.loc[sample, feature] for _, x in enumerate(preds)]
                test_std.loc[sample, feature] = np.std(predictions)
                predictions_norm = [np.abs(pred - prediction) for pred in predictions]
                test_val = test.loc[sample, feature]
                test_valn = np.abs(test_val - prediction)
                pc = len(np.where(predictions_norm >= test_valn)[0]) / len(predictions_norm)
                test_pc.loc[sample, feature] = pc

                file.write("{},{},{},{}".format(sample, feature, prediction,
                    test_pc.loc[sample, feature]))
                pred_text = ""
                predn_text = ""
                for pred, predn in list(zip(predictions, predictions_norm)):
                    pred_text += ",{}".format(pred)
                    predn_text += ",{}".format(predn)

                file.write(",{}{}".format(test_val, pred_text))
                file.write(",{}{}".format(test_valn, predn_text))
                file.write("\n")

        file.close()

        test_pc_vec = test_pc.values.reshape(test_pc.shape[0] * test_pc.shape[1])
        test_pc_sort_idx = np.argsort(test_pc_vec)

        test_pc_pos = [(int(x/(test_pc.shape[1])),x % test_pc.shape[1])
            for x in test_pc_sort_idx]

        df = pd.DataFrame(index=pd.Index([], name="ID"),
            columns=["Sample Index", "Column","Row Index","Column Index","Component Index",
            "Prediction", "more extreme bagging estimators",
            "Input", "Standard Deviations"])
        for (row_idx, col_idx) in test_pc_pos:
            std = np.abs((pred_bagging.iloc[row_idx, col_idx] -
                test.iloc[row_idx, col_idx]) / test_std.iloc[row_idx, col_idx]) if \
                test_std.iloc[row_idx, col_idx] != 0 else np.inf
            sample_idx = test.index[row_idx]
            id = "{}_{}".format(sample_idx, features[col_idx])
            df.loc[id,:] = [sample_idx, features[col_idx], row_idx,
                col_idx, "", pred_bagging.iloc[row_idx, col_idx],
                test_pc.iloc[row_idx, col_idx], test.iloc[row_idx, col_idx],
                std
            ]

        report = df.sort_values(
            by=["more extreme bagging estimators", "Standard Deviations"],
            ascending=[True, False]
        )
        report.to_csv("outliers_basic.csv")

        return pred_bagging, test_std, test_pc, report


    def predictAllEstimators(self, test):
        print("DatasetAnalyzer.predictAllEstimators")
        n = 0
        features = BasicAnalyzer.train.getFeatures()
        for feature in features:
            n = np.max([n,
                self.gcvs["models"][feature].best_estimator_['bagging'].n_estimators])

        print("n: {}".format(n))
        preds = [test.copy() for k in range(n)]
        pred_bagging = test.copy()
        for feature in features:
            feats = test.getFeaturesWithout([feature])
            estimator = self.gcvs["models"][feature].best_estimator_
            pred_bagging[feature] = estimator.predict(test[feats].values)
            for i, estimator_ in enumerate(estimator['bagging'].estimators_):
                pipeline = Pipeline([
                    ('features', estimator['features']),
                    ('bagging', estimator_),
                ])
                preds[i][feature] = pipeline.predict(test[feats].values)

        return pred_bagging, preds


    @staticmethod
    def printInfo():
        ''' Prints some basic information about the dataset '''

        text = ""
        text += BasicAnalyzer.printDimensions()
        text += BasicAnalyzer.printMissingValues()
        text += BasicAnalyzer.printBasicStats()
        return text


    @staticmethod
    def printDimensions(test=None):
        ''' Prints the number of samples and features in the dataset '''

        text = "DIMENSIONS\n"
        text += "  Dataset:                "
        text += "      TRAIN" if BasicAnalyzer.train is not None else ""
        text += "      VALID" if BasicAnalyzer.valid is not None else ""
        text += "       TEST" if test is not None else ""
        text += "\n"
        text += "  Features:               "
        text += " {:10d}".format(BasicAnalyzer.train.shape[1]) if BasicAnalyzer.train is not None else ""
        text += " {:10d}".format(BasicAnalyzer.valid.shape[1]) if BasicAnalyzer.valid is not None else ""
        text += " {:10d}".format(test.shape[1]) if test is not None else ""
        text += "\n"
        text += "  Samples:                "
        text += " {:10d}".format(BasicAnalyzer.train.shape[0]) if BasicAnalyzer.train is not None else ""
        text += " {:10d}".format(BasicAnalyzer.valid.shape[0]) if BasicAnalyzer.valid is not None else ""
        text += " {:10d}".format(test.shape[0]) if test is not None else ""
        text += "\n\n"
        print(text)
        return text


    @staticmethod
    def printMissingValues(test=None):
        ''' Prints the number of missing values in the dataset '''

        text = "MISSING VALUES\n"
        text += "  Dataset:                "
        text += "      TRAIN" if BasicAnalyzer.train is not None else ""
        text += "      VALID" if BasicAnalyzer.valid is not None else ""
        text += "       TEST" if test is not None else ""
        text += "\n"
        for feature in BasicAnalyzer.features_max:
            text += "  {:23s} ".format(feature + ":")
            text += " {:10d}".format(len(BasicAnalyzer.train.nans_idx[feature])) if BasicAnalyzer.train is not None else ""
            text += " {:10d}".format(len(BasicAnalyzer.valid.nans_idx[feature])) if BasicAnalyzer.valid is not None else ""
            text += " {:10d}".format(len(test.nans_idx[feature])) if test is not None else ""
            text += "\n"
        print(text)
        return text

    @staticmethod
    def printBasicStats(test=None):
        '''
        Prints the average value and the standard deviation for each numeric feature.
        The function just considers the entries listed in the "XYZ_filled_idx"
        '''

        text = "BASIC STATS\n"
        text += "  Dataset:                "
        text += "      TRAIN" if BasicAnalyzer.train is not None else ""
        text += "      VALID" if BasicAnalyzer.valid is not None else ""
        text += "       TEST" if test is not None else ""
        text += "\n"
        text += "  AVG\n"
        for feature in BasicAnalyzer.features_max:
            text += "    {:21s} ".format(feature + ":")

            for ds in [BasicAnalyzer.train, BasicAnalyzer.valid, test]:
                if ds is not None:
                    if (np.issubdtype(ds[feature].dtype, np.number)):
                        text += " {:10.2f}".format(
                            np.average(ds.getFilled(feature, [feature])))
                    else:
                        text += "           "
            text += "\n"

        text += "  STANDARD DEVIATION\n"
        for feature in BasicAnalyzer.features_max:
            text += "    {:21s} ".format(feature + ":")

            for ds in [BasicAnalyzer.train, BasicAnalyzer.valid, test]:
                if ds is not None:
                    if (np.issubdtype(ds[feature].dtype, np.number)):
                        text += " {:10.2f}".format(
                            np.std(ds.getFilled(feature, [feature]))[0])
                    else:
                        text += "        ---"
            text += "\n"
        print(text)
        return text


    def printFeatureModelsFit(self, test=None, features=None):
        '''
        Prints key quality numbers for the fit of the Missing Values Models
        '''

        text = "MISSING VALUES MODEL - FIT"
        if features == None:
            features = BasicAnalyzer.train.getFeatures()

        text += "  SMSE:                               TRAIN      VALID       TEST"
        for feature in features:
            text += "  {:18s} ".format(feature)
            mse_train, mse_train_avg = self.printFeatureModelFitSMSE(BasicAnalyzer.train, feature)
            mse_valid, mse_valid_avg = self.printFeatureModelFitSMSE(BasicAnalyzer.valid, feature)
            mse_test, mse_test_avg = self.printFeatureModelFitSMSE(test, feature)

            line_est = "    {:25s} ".format(self.getEstimatorShortDescription(feature))
            line_est += " {:10.2f}".format(mse_train) if mse_train else " ----------"
            line_est += " {:10.2f}".format(mse_valid) if mse_valid else " ----------"
            line_est += " {:10.2f}".format(mse_test) if mse_test else " ----------"

            line_avg = "    vs. mean                  "
            line_avg += " {:10.2f}".format(mse_train_avg) if mse_train_avg else " ----------"
            line_avg += " {:10.2f}".format(mse_valid_avg) if mse_valid_avg else " ----------"
            line_avg += " {:10.2f}".format(mse_test_avg) if mse_test_avg else " ----------"

            text += line_est
            text += line_avg

        print(text)
        return text


    def printFeatureModelFitSMSE(self, ds, feature):
        if feature in self.gcvs["models"].keys() and ds.shape[0] > 0:
            pred = self.gcvs["models"][feature].best_estimator_.predict(
                ds.getFilled(feature, [feature], False)
            )
            mse = mean_squared_error(ds.getFilled(feature, [feature]), pred)
            mse_mean = mean_squared_error(
                ds.getFilled(feature, [feature]),
                [np.average(ds.getFilled(feature, [feature]))]*ds.getFilled(feature).shape[0]
            )
            return np.sqrt(mse), np.sqrt(mse_mean)
        return None, None


    def print_residuals(self, test):
        print("BasicAnalyzer.print_residuals")
        features = BasicAnalyzer.train.getFeatures()
        
        text = "RESIDUALS\n\n"
        text += "AVG and Standard Deviation\n\n"
        text += "{:<20s} {:>20s}  {:20s} ".format("Feature", "TRAIN", "")
        text += " {:>20s}  {:20s} ".format("TEST", "") if test is not None else ""
        
        datasets = [BasicAnalyzer.train]
        datasets += [test] if test is not None else []

        residuals = []
        for ds in datasets:
            ds_ = Dataset(ds.name, ds.type, ds.loc[ds.all_filled_idx, :].copy())               
            pred_bagging, preds = self.predictAllEstimators(ds_)
            preds = [pred_bagging] + preds
            
            res = []
            for i, pred in enumerate(preds):
                res_ = pred - ds_
                res += [res_]
                estimator_name = i - 1 if i > 0 else "Bagging"
                res_.to_csv("debug/residuals_{}_{}.csv".format(ds.type, estimator_name))
                pred.to_csv("debug/predicted_{}_{}.csv".format(ds.type, estimator_name))
                
            ds_.to_csv("debug/actual_{}.csv".format(ds.type))

            residuals += [res]

        # calculates the correlation between the various feature estimators
        textcorr = "\n\nAverage Correlations between estimators' residuals"
        textcorr += "\n{:<20s}  {:>20s}".format("Feature", "TRAIN")
        textcorr += " {:>20s} ".format("TEST") if test is not None else ""
        for f, feature in enumerate(features):
            textcorr += "\n{:<20s}:".format(feature)
            for d, ds in enumerate(datasets):
                res = residuals[d]
                feature_df = pd.DataFrame([x[feature] for x in res])
                feature_df = feature_df.transpose()
                feature_df = feature_df.corr()
                feature_df.to_csv("debug/residuals_corr_{}_{}.csv".format(ds.type, f))
                textcorr += " {:20.4f}".format(np.average(feature_df))

        # prepares the output
        texts = [text]
        residuals_all_ds = list(zip(*residuals))
        for i, dsets in enumerate(residuals_all_ds):
            estimator_name = i - 1 if i > 0 else "Bagging"
            text = "\n\n\nEstimator: {}\n\n".format(estimator_name)
            for feature in features:
                text += "\n{:<20s}".format(feature)
                for ds in dsets:
                    text += " {:>20.4f}".format(np.average(ds.loc[:,feature].astype('float32')))
                    text += " ({:<.4f})".format(np.std(ds.loc[:,feature].astype('float32')))               
            texts += [text]
                
        text = texts[0] + texts[1]
        text += textcorr + "\n\n\n\n" + texts[0]
        for t in texts[2:]:
            text += t
            
        print(text)
        return text


    def printImputedVsActual(self, df, df_imp, features=None, showheader=True):
        if features == None:
            features = BasicAnalyzer.train.getFeatures()

        text = ""
        if showheader:
            text += "\n\n{:<25s} {:>20s} {:>20s} ({:<s})".format(
                "Standard Error", "Model", "vs np.average", "R^2")
        for feature in features:
            idxs = df_imp.nans_idx[feature]
            act = df.loc[idxs, feature]
            pred = df_imp.loc[idxs, feature]

            mse = mean_squared_error(act, pred)
            if len(idxs) < df_imp.shape[0]:
                mse_mean = mean_squared_error(act,
                    [np.average(df_imp.getFilled(feature,
                        [feature]))]*len(act))
                text += "\n{:<25s} {:>20.2f} {:>20.2f} ({:<.2f})".format(
                    feature, mse**0.5, mse_mean**0.5, 1 - (mse / mse_mean))
            else:
                text += "\n{:<25s} {:>20.2f}".format(feature, mse**0.5)

        print(text)
        return text

    def getFeatureModelShortDescription(self, feature):
        ''' returns the class of the bagged estimator '''

        return self.gcvs["models"][feature].best_estimator_['bagging']. \
            base_estimator.__class__.__name__


    def getFeatureModelDescription(self, feature):
        ''' returs the best parameters for the feature prediction '''

        description = "{} Model:\n".format(feature)
        description += self.gcvs["models"][feature].best_estimator_['bagging']. \
            base_estimator.__class__.__name__
        description += "\n"
        description += str(self.gcvs["models"][feature].best_params_)
        description += "\n\n"
        return description


    def printFeatureModelDescriptions(self, features=None):
        ''' prints the Estimator Description for all feature models '''

        text = ""
        if features == None:
            features = BasicAnalyzer.train.getFeatures()

        for feature in features:
            text += self.getFeatureModelDescription(feature)

        print(text)
        return text


    def plotImputedVsActual(self, df, df_imp, features=None):
        for feature in features:
            idxs = df_imp.nans_idx[feature]
            act = df.loc[idxs, feature]
            pred = df_imp.loc[idxs, feature]

            plt.style.use('seaborn')
            fig = plt.figure(figsize=(5,5))
            plt.scatter(act, pred)
            plt.show()


    def plotPredictedVsActual(self, df, features=None):
        for feature in features:
            pred = self.gcvs["models"][feature].best_estimator_.predict(
                df.getFilled(feature, [feature], False)
            )
            act = df.getFilled(feature, [feature])

            plt.style.use('seaborn')
            fig = plt.figure(figsize=(5,5))
            plt.scatter(act, pred)
            plt.show()


    def saveImputedAlchemiteFormat(self, filename):
        self.imputed.to_csv(filename, index=False, header=False, line_terminator='\n')
