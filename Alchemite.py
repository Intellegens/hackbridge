from DatasetAnalyzer import Dataset, BasicAnalyzer
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


        data = test.to_csv(line_terminator="\n")
        self.dataset_outliers = client.Dataset(
            name = test.name,
            row_count = test.shape[0],
            column_headers = BasicAnalyzer.train.getFeatures(),
            descriptor_columns = [0]*len(BasicAnalyzer.train.getFeatures()),
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



        # return pred_bagging, test_std, test_pc
        return report
