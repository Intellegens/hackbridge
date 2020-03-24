from DatasetAnalyzer import Dataset, BasicAnalyzer
import numpy as np
import pandas as pd

from io import StringIO
import time
import alchemite_apiclient as client
from alchemite_apiclient.extensions import Configuration
import itertools


def chunks(ds, index=False, chunk_size=1000):
    """
    Generator which splits dataset into chunks with chunk_size rows each.
    The first row of the file (the column header row) is prepended to the top
    of each chunk.

    Parameters
    ----------
    file_name : str
    Path to file which is to be split into chunks
    chunk_size : int
    How many rows (excluding the column header row) there should be in each
    chunk.  The last chunk may have less than this.

    Yields
    ------
    str
    A chunk of file_name with the column header row prepended to the top of
    each file.
    """

    file = StringIO(ds.to_csv(index=index, line_terminator="\n"))

    column_header_row = file.readline()

    while True:
        chunk = list(itertools.islice(file, chunk_size))
        if chunk == []:
        	# Reached the end of the file
        	break

        yield column_header_row + ''.join(chunk)



class AlchemiteAnalyzer(BasicAnalyzer):
    def __init__(self, train=None, valid=None, credentials='credentials.json'):
        super().__init__(train, valid)

        self.configuration = Configuration()
        self.api_default = client.DefaultApi(client.ApiClient(self.configuration))
        self.api_models = client.ModelsApi(client.ApiClient(self.configuration))
        self.api_datasets = client.DatasetsApi(client.ApiClient(self.configuration))
        self.configuration.credentials = credentials
        api_response = self.api_default.version_get()
        print(api_response)


    def close(self):
        self.api_models.models_id_delete(self.model_id)
        self.api_datasets.datasets_id_delete(BasicAnalyzer.train.dataset_id)


    def fitFeatureModels(self, features=None):
        print("Alchemite.fitFeatureModels")
        data = BasicAnalyzer.train.to_csv(line_terminator="\n")
        self.dataset = client.Dataset(
            name = BasicAnalyzer.train.name,
            row_count = BasicAnalyzer.train.shape[0],
            column_headers = BasicAnalyzer.train.getFeatures(),
            descriptor_columns = [0]*len(BasicAnalyzer.train.getFeatures()),
            # data = data
        )

        BasicAnalyzer.train.dataset_id = self.api_datasets.datasets_post(dataset=self.dataset)
        for chunk_number, chunk in enumerate(chunks(BasicAnalyzer.train, 50)):
            response = self.api_datasets.datasets_id_chunks_chunk_number_put(
                BasicAnalyzer.train.dataset_id, chunk_number, body=chunk)
            print('Uploaded chunk', chunk_number)

        self.api_datasets.datasets_id_uploaded_post(BasicAnalyzer.train.dataset_id)
        print('Collated dataset')

        print('--- dataset metadata ---')
        print(self.api_datasets.datasets_id_get(BasicAnalyzer.train.dataset_id))

        BasicAnalyzer.train.dataset_metadata = self.api_datasets.datasets_id_get(BasicAnalyzer.train.dataset_id)

        self.model = client.Model(
            name = BasicAnalyzer.train.name,
            training_method = 'alchemite',
            training_dataset_id = BasicAnalyzer.train.dataset_id
        )

        self.model_id = self.api_models.models_post(model=self.model)
        self.model_metadata = self.api_models.models_id_get(self.model_id)

        BasicAnalyzer.train_request = client.TrainRequest(
            validation = '80/20',
            hyperparameter_optimization = 'none',
            virtual_experiment_validation = False,
        )

        self.response = self.api_models.models_id_train_put(self.model_id,
            train_request=BasicAnalyzer.train_request)

        t0 = time.time()
        while True:
        	self.model_metadata = self.api_models.models_id_get(self.model_id)
        	print(
        		'Time:', time.time()-t0,
        		'Optimization progress:', self.model_metadata.hyperparameter_optimization_progress,
        		'\tTraining progress:', self.model_metadata.training_progress
        	)
        	if self.model_metadata.status == 'trained' or \
               self.model_metadata.status == 'failed':
        		break
        	time.sleep(5)
        print('Training time:', time.time()-t0)


    def iterateMissingValuePredictions(self, df, features=None):
        print("Alchemite.iterateMissingValuePredictions")
        if features == None:
            features = BasicAnalyzer.train.getFeatures()

        df.restoreMissingValues()
        data = df.to_csv(index=False, line_terminator="\n")
        self.impute_request = client.ImputeRequest(
            return_probability_distribution = False,
            data = data,
        )

        print("Alchemite.iterateMissingValuePredictions - api_models.models_id_impute_put")
        self.response = self.api_models.models_id_impute_put(
            self.model_id, impute_request=self.impute_request)

        txt = ""
        for meaning in ["orig", "imputed", "std"]:
            for col in features:
                txt += "{}_{},".format(meaning, col)
        txt = txt[0:len(txt)-1] + "\n"

        DATA = StringIO(txt + self.response)

        self.imputed = pd.read_csv(DATA, sep=",")
        self.imputed.index = list(df.index)

        for feature in features:
            for idx in df.nans_idx[feature]:
                col = "imputed_{}".format(feature)
                val = self.imputed.loc[idx, col]
                df.loc[idx, feature] = val


    def findOutliers(self, test):
        print("Alchemite.findOutliers")

        features = BasicAnalyzer.train.getFeatures()

        data = test.to_csv(line_terminator="\n")
        self.dataset_outliers = client.Dataset(
            name = test.name,
            row_count = test.shape[0],
            column_headers = features,
            descriptor_columns = [0]*len(features),
            # data = data
        )

        test.dataset_id = self.api_datasets.datasets_post(dataset=self.dataset_outliers)
        for chunk_number, chunk in enumerate(chunks(test, 50)):
            response = self.api_datasets.datasets_id_chunks_chunk_number_put(
                test.dataset_id, chunk_number, body=chunk)
            print('Uploaded chunk', chunk_number)

        self.api_datasets.datasets_id_uploaded_post(test.dataset_id)
        print('Collated dataset')

        print('--- dataset metadata ---')
        print(self.api_datasets.datasets_id_get(test.dataset_id))

        test.dataset_metadata = self.api_datasets.datasets_id_get(test.dataset_id)



        test_pc = Dataset(test.name, "outlier_pc", test.loc[test.all_filled_idx, :].copy())
        test_std = Dataset(test.name, "outlier_pc", test.loc[test.all_filled_idx, :].copy())

        outliers_request = client.OutliersRequest(dataset_id=test.dataset_id)
        response = self.api_models.models_id_outliers_put(
            self.model_id, outliers_request=outliers_request, _preload_content=False
        )

        DATA = StringIO(str(response.data.decode()))
        report = pd.read_csv(DATA, sep=",", lineterminator='\n')

        idx = report["Row Index"] - 1
        report["Sample Index"] = test.index[idx]

        report.to_csv("outliers_alchemite.csv")


        test_avg = Dataset(test.name, "outlier_pc", test.loc[test.all_filled_idx, :].copy())
        test_pc = Dataset(test.name, "outlier_pc", test.loc[test.all_filled_idx, :].copy())
        test_std = Dataset(test.name, "outlier_pc", test.loc[test.all_filled_idx, :].copy())

        file = open("debug\\findOutliers_alchemite.csv", "w")
        file.write("{},{},{},{},{}\n".format("sample", "feature", "prediction", "more extreme", "actual value", "predictions"))

        for f, feature in enumerate(features):
            test_without_feature = Dataset(test.name, "outlier_wo_feature", test.loc[test.all_filled_idx, :].copy())
            test_without_feature.loc[:, feature] = None

            data = test_without_feature.to_csv(index=False, line_terminator="\n")

            impute_request = client.ImputeRequest(
                return_probability_distribution = True,
                data = data,
            )

            response = self.api_models.models_id_impute_put(self.model_id,
                impute_request=impute_request)

            responseIO = StringIO(response)
            responseDF = pd.read_csv(responseIO, header=None, sep=",", lineterminator='\n')
            responseDF.columns = features + [feature + "_est" for feature in features]
            responseDF.index = test_without_feature.index

            for sample in test_without_feature.index:
                predictions_split = responseDF.loc[sample, feature + "_est"]
                predictions_split = predictions_split.split('#')
                predictions = [float(x) for x in predictions_split]
                prediction = np.average(predictions)
                test_std.loc[sample, feature] = np.std(predictions)
                predictions_norm = [np.abs(pred - prediction) for pred in predictions]
                test_val = test.loc[sample, feature]
                test_valn = np.abs(test_val - prediction)
                pc = len(np.where(predictions_norm >= test_valn)[0]) / len(predictions_norm)
                test_pc.loc[sample, feature] = pc
                test_avg.loc[sample, feature] = prediction

                file.write("{},{},{},{},{}".format(sample, feature,
                    test_avg.loc[sample, feature], test_pc.loc[sample, feature],
                    test_val))
                for pred in predictions:
                    file.write(",{}".format(pred))
                file.write("\n")
            # input('input something!: ')

        file.close()
        test_avg.to_csv("debug\\findOutliers_alchemite_pred_avg.csv")

        # return pred_bagging, test_std, test_pc
        return test_avg, test_std, test_pc, report


    def predictAllEstimators(self, test):
        print("Alchemite.predictAllEstimators")
        features = BasicAnalyzer.train.getFeatures()

        preds = []
        preds_cnt = -1
        pred_bagging = test.copy()
        for feature in features:
            test_without_feature = Dataset(test.name, "outlier_wo_feature", test.loc[test.all_filled_idx, :].copy())
            test_without_feature.loc[:, feature] = None

            data = test_without_feature.to_csv(index=False, line_terminator="\n")

            impute_request = client.ImputeRequest(
                return_probability_distribution = True,
                data = data,
            )

            response = self.api_models.models_id_impute_put(self.model_id,
                impute_request=impute_request)

            responseIO = StringIO(response)
            responseDF = pd.read_csv(responseIO, header=None, sep=",", lineterminator='\n')
            responseDF.columns = features + [feature + "_est" for feature in features]
            responseDF.index = test_without_feature.index

            for sample in test_without_feature.index:
                predictions_split = responseDF.loc[sample, feature + "_est"]
                predictions_split = predictions_split.split('#')
                predictions = [float(x) for x in predictions_split]
                
                prediction = np.average(predictions)
                pred_bagging.loc[sample, feature] = prediction

                for p, pred in enumerate(predictions):
                    if p > preds_cnt:
                        h = np.zeros(test.shape)
                        h = pd.DataFrame(h)
                        h.index = test.index
                        h.columns = test.columns
                        h.loc[:,:] = None
                        preds += [h]
                        preds_cnt += 1
                    preds[p].loc[sample, feature] = pred

        return pred_bagging, preds