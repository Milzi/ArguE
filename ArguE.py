from ArgumentationClassifier import DataLoader as dl
from ArgumentationClassifier import ADSClassifier
from ArgumentationClassifier import FeatureExtractor as af
import pandas as pd
from keras import models
from sklearn.externals import joblib



"""Interface for loading the data from the ArguE-XML-File format,
    extracting the features and training and testing the classifiers."""
class ArguE:


    def __init__(self):

        self.classifier = ADSClassifier
        self.featureExtractor = None

    def load_NN(self, model=None):
        """Load neural network from h5.file"""

        if model != None:
            return models.load_model(model)


    def load_RNN(self, model=None):
        """Load recurrent neural network from h5.file"""

        if model != None:
            return models.load_model(model)



    def load_RF_OR(self, model=None):
        """Load randomForest or OneR classifier from pickle"""

        if model != None:
            return joblib.load(model)

    def train_NN_classifier(self, trainingData, epochs=500, saveModel=None):
        """train neural network classifier
            parameter:
            trainingData: training-data
            epochs: nuumber of epocs, default = 500
            saveModel: directory to h5 file for saving the model
        """

        trainedNN = self.classifier.train_NN_classifier(trainingData, epochs)

        if saveModel != None:

            trainedNN.save(saveModel)

        return trainedNN

    def train_RNN_classifier(self, trainingData, epochs=500, saveModel=None):
        """train recurrent neural network classifier
            parameter:
            trainingData: training-data
            epochs: nuumber of epocs, default = 500
            saveModel: directory to h5 file for saving the model
        """

        trainedRNN = self.classifier.train_RNN_classifier(trainingData, epochs)

        if saveModel != None:

            trainedRNN.save(saveModel)

        return trainedRNN

    def train_RF_classifier(self, trainingData, saveModel=None):
        """train random forest classifier
            parameter:
            trainingData: training-data
            saveModel: directory to h5 file for saving the model
        """

        randomForest = self.classifier.train_RF_classifier(trainingData)

        if saveModel != None:
            joblib.dump(randomForest, saveModel)

        return randomForest

    def train_Dummy_classifier(self, trainingData, saveModel=None):
        """train dummy classifier
            parameter:
            trainingData: training-data
            saveModel: directory to h5 file for saving the model
        """

        zeroR = self.classifier.train_Dummy_classifier(trainingData)

        if saveModel != None:
            joblib.dump(zeroR, saveModel)

        return zeroR

    def test_classifier(self, testData, classifier):
        """test classifier
            parameter:
            testData: testing-data
            classifier: classifier model that should be tested
        """

        self.classifier.test_classifier(testData, classifier)

    def get_baseline_results(self, dataset):
        """prints baseline results using 50-50 guessing
            parameter:
            dataset: testing-data
        """
        self.classifier.baseline_results(dataset)

    def load_Data(self, dataDirectory=None, store=None):
        """load the data from XML-file and automatically extracts the features
            parameter:
            dataDirectory: directory of the XML-file
            store: directory to the h5.file for storing the loaded data and features
        """

        data = dl.loadData(dataDirectory)
        dataset = self.generate_Features(data)

        if store != None:
            dataStore = pd.HDFStore(store)
            dataStore['dataset'] = data
            dataStore['feature'] = dataset

        return dataset

    def generate_Features(self, dataset):
        """extracts the features of the passed pandas dataframe
            parameter:
            dataset: pandas dataframe containing loaded data
        """

        if self.featureExtractor == None:
            self.featureExtractor = af.AdvancedFeatureExtractor()
        return self.featureExtractor.extractFeatures(dataset)

    def save_data(self, dataset, store):
        """save dataframe to store (h5.file)
            parameter:
            dataset: pandas dataframe containing data
            store: directory to h5.file for storage
        """

        dataStore = pd.HDFStore(store)
        dataStore["feature"] = dataset


    def load_Data_From_Store(self, store):
        """load data and features from h5.file
            parameter:
            store: directory to h5.file containing the data or features
        """

        dataset = None

        dataStore = pd.HDFStore(store)

        try:

            dataset = dataStore["feature"]
            print("-----LOAD FEATURES FROM STORE-----")

        except (KeyError, IOError)  as e:

            print("-----NO FEATURES FOUND AT PASSED H5 FILE-----")

            try:

                dataset = dataStore["dataset"]
                print("-----LOAD DATASET FROM STORE-----")
                dataset = self.generate_Features(dataset)
                dataStore["feature"] = dataset

            except (KeyError, IOError)  as e:

                print("-----NO DATA FOUND AT PASSED H5 FILE-----")

        return dataset

    def split_data(self, dataset, splitting=0.1):
        """splits data into training and testset
            parameter:
            dataset: pandas dataframe containing the training data
            splitting: percentage split of the data 0.1 = 10% testing data
            return values: trainset, testset
        """

        train, test = dl.supervised_split(dataset, splitting)

        return train, test

    def balance_data(self, dataset, balancing=0.5):
        """balances the passed data
            parameter:
            dataset: pandas dataframe containing the data
            balancing: precentage of the balancing -> 0.5 = equal 50-50 balncing
        """

        dataset = dl.balance_dataset(dataset, balancing)

        return dataset

    def show_dataset_statistics(self):
        """print the statistics of the datasets"""

        print("AraucariaDB")
        dl.loadStatistics("resources/corpora/araucaria")
        print("microtext")
        dl.loadStatistics("resources/corpora/microtext")
        print("rrd")
        dl.loadStatistics("resources/corpora/rrd")
        print("schemes")
        dl.loadStatistics("resources/corpora/schemes")
        print("STAB")
        dl.loadStatistics("resources/corpora/studentEssays")
        print("IBM")
        dl.loadStatistics("resources/corpora/ibm")
        print("ArguE")
        dl.loadStatistics("resources/corpora/arguE")

    def change_labels(self, dataset, attack=False, bidirect=True):
        """changes the labels depending on the classification task
            parameter:
            dataset: pandas dataframe containing the data
            attack: if true, the attack label will remain in the dataset, default is false
            bidirect: allow bidirectional relation instead of one directional
        """

        if not attack:
            dataset.loc[dataset.label == 2, 'label'] = 1
            dataset.loc[dataset.label == -2, 'label'] = -1

        if bidirect:
            dataset.loc[dataset.label == -1, 'label'] = 1
        else:
            dataset.loc[dataset.label == -1, 'label'] = 0

        return dataset

class main:

    aif = "resources/datasets/aif.h5"
    se = "resources/datasets/se.h5"
    ibm = "resources/datasets/ibm.h5"
    argu = "resources/datasets/arguE.h5"

    aifTrain = "resources/datasets/training/aifTrain.h5"
    aifTest = "resources/datasets/testing/aifTest.h5"
    seTrain = "resources/datasets/training/seTrain.h5"
    seTest = "resources/datasets/testing/seTest.h5"
    ibmTrain = "resources/datasets/training/ibmTrain.h5"
    ibmTest = "resources/datasets/testing/ibmTest.h5"
    argueTrain = "resources/datasets/training/argueTrain.h5"
    argueTest = "resources/datasets/testing/argueTest.h5"

    arguE = ArguE()

    #######################################################################

    ####### Training #######

    print("################## TRAINING:")

    #data is already balanced and labels are changed
    trainSet = arguE.load_Data_From_Store(aifTrain)

    OneR = arguE.train_Dummy_classifier(trainSet, "resources/classifierModels/aif_or.pkl")
    RF = arguE.train_RF_classifier(trainSet, "resources/classifierModels/all_rf.pkl")
    RNN = arguE.train_RNN_classifier(trainSet, epochs=25, saveModel="resources/classifierModels/ibm_rnn.h5")

    ####### Testing #######

    print("################## TESTING:")

    testSet = arguE.load_Data_From_Store(aifTest)

    arguE.test_classifier(testSet, OneR)

    arguE.test_classifier(testSet, RF)

    arguE.test_classifier(testSet, RNN)

    #######################################################################
