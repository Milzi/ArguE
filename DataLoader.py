import os
import xmltodict
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None

def loadData(directory):

    print("-----LOAD DATASET FROM XML-FILE-----")
    dataset = loadDataset(directory)
    print("-----DATASET LOADED-----")
    print("-----EXTRACTED DATA SIZE-----")
    print("-----SIZE: " +str(len(dataset)) + "-----")

    return dataset

def loadStatistics(directory):

    conclusionCount = 0
    claimCount=0
    premiseCount=0
    supportCount = 0
    attackCount = 0
    listTyp = list().__class__

    for e, annotationFile in enumerate(os.listdir(directory)):

        annotationFilePath = os.path.join(directory, annotationFile)

        with open (annotationFilePath, "r") as myfile:
            data = myfile.read()

        #print("Reading File: " + annotationFilePath)
        xmlData = xmltodict.parse(data)

        propositions = xmlData["Annotation"]["Proposition"]

        for propositionIndex in range(len(propositions)):

            if propositions[propositionIndex]["ADU"]["@type"] == "conclusion":

                conclusionCount += 1

            elif propositions[propositionIndex]["ADU"]["@type"] == "claim":

                claimCount += 1

                if "Relation" in propositions[propositionIndex].keys():

                    if propositions[propositionIndex]["Relation"].__class__ == listTyp:

                        for relation in propositions[propositionIndex]["Relation"]:

                            if relation["@typeBinary"] == "0":
                                supportCount += 1
                            elif relation["@typeBinary"] == "1":
                                attackCount += 1
                    else:

                        if propositions[propositionIndex]["Relation"]["@typeBinary"] == "0":
                                supportCount += 1
                        elif propositions[propositionIndex]["Relation"]["@typeBinary"] == "1":
                                attackCount += 1

            elif propositions[propositionIndex]["ADU"]["@type"] == "premise":

                premiseCount += 1

                if "Relation" in propositions[propositionIndex].keys():

                    if propositions[propositionIndex]["Relation"].__class__ == listTyp:

                        for relation in propositions[propositionIndex]["Relation"]:

                            if relation["@typeBinary"] == "0":
                                supportCount += 1
                            elif relation["@typeBinary"] == "1":
                                attackCount += 1
                    else:

                        if propositions[propositionIndex]["Relation"]["@typeBinary"] == "0":
                                supportCount += 1
                        elif propositions[propositionIndex]["Relation"]["@typeBinary"] == "1":
                                attackCount += 1

    print("-----------------")
    print("CONCLUSIONS:" + str(conclusionCount))
    print("CLAIM:" + str(claimCount))
    print("PREMISE:" + str(premiseCount))
    print("SUPPORT:" + str(supportCount))
    print("ATTACK:" + str(attackCount))



def loadDataset(directory):

    listTyp = list().__class__
    inputDict = list()

    for e, annotationFile in enumerate(os.listdir(directory)):

        relationMatrix = {}
        annotationFilePath = os.path.join(directory, annotationFile)

        with open (annotationFilePath, "r") as myfile:
            data = myfile.read()

        print("Reading File: " + annotationFilePath)
        xmlData = xmltodict.parse(data)

        argumentationID = e

        matrixLength = len(xmlData["Annotation"]["Proposition"])
        relationCount = 0
        totalRelation = matrixLength*matrixLength
        relationMatrix = (matrixLength,matrixLength)
        relationMatrix=np.zeros(relationMatrix)

        propositions = xmlData["Annotation"]["Proposition"]

        for propositionIndex in range(len(propositions)):

            currentProposition = propositions[propositionIndex]

            if currentProposition["ADU"]["@type"] != "conclusion" and "Relation" in currentProposition.keys():

                partnerList = list()
                relationTypeList = list()

                if currentProposition["Relation"].__class__ == listTyp:

                    for relation in range(len(currentProposition["Relation"])):

                        partnerList.append(currentProposition["Relation"][relation]["@partnerID"])
                        relationTypeList.append(currentProposition["Relation"][relation]["@typeBinary"])

                else:

                    partnerList.append(currentProposition["Relation"]["@partnerID"])
                    relationTypeList.append(currentProposition["Relation"]["@typeBinary"])


                for partnerIndex in range(len(partnerList)):

                    for secondPropositionIndex in range(len(propositions)):

                        if partnerList[partnerIndex] == propositions[secondPropositionIndex]["@id"]:

                            if relationTypeList[partnerIndex] == "0":

                               relationMatrix[propositionIndex][secondPropositionIndex] = 1

                               relationMatrix[secondPropositionIndex][propositionIndex] = -1

                            elif relationTypeList[partnerIndex] == "1":

                                relationMatrix[propositionIndex][secondPropositionIndex] = 2

                                relationMatrix[secondPropositionIndex][propositionIndex] = -2

                            else:

                                relationMatrix[propositionIndex][secondPropositionIndex] = -3

        for i in range(len(relationMatrix)):

            for j in range(len(relationMatrix[i])):

                if i != j and relationMatrix[i][j] > -3:

                    proposition1= propositions[i]["text"]
                    proposition2= propositions[j]["text"]

                    if isTooLong(proposition1) or isTooLong(proposition2):
                        continue

                    originalSentenceArg1 = propositions[i]["text"]
                    originalSentenceArg2 = propositions[j]["text"]

                    if "TextPosition" in propositions[i].keys():

                        if propositions[i]["TextPosition"]["@start"] != "-1":

                            sent_tokenize_list = sent_tokenize(xmlData["Annotation"]["OriginalText"])

                            for sentence in sent_tokenize_list:

                                if propositions[i]["text"] in sentence:

                                    originalSentenceArg1 = sentence

                        if propositions[j]["TextPosition"]["@start"] != "-1":

                            sent_tokenize_list = sent_tokenize(xmlData["Annotation"]["OriginalText"])

                            for sentence in sent_tokenize_list:

                                if propositions[j]["text"] in sentence:

                                    originalSentenceArg2 = sentence

                    inputDict.append({'argumentationID':argumentationID, 'arg1': propositions[i]["text"], 'originalArg1': originalSentenceArg1, 'arg2': propositions[j]["text"], 'originalArg2': originalSentenceArg2, 'label':relationMatrix[i][j]})

    dataFrame = pd.DataFrame.from_dict(inputDict, orient='columns')

    return(dataFrame)

def isTooLong( proposition):

    if len(sent_tokenize(proposition)) > 1:
        return True

    elif len(word_tokenize(proposition))>30:
        return True

    else:
        return False


def balance_dataset(dataset, balanceRatio):

    RELATION_RATIO = balanceRatio

    labelMatrix = dataset.as_matrix(columns=['label'])
    numberOfRelations = np.count_nonzero(labelMatrix)
    relationRatio = numberOfRelations/len(dataset)

    if relationRatio < RELATION_RATIO:

        print("-----DATA IS UNBALANCED CURRENT SIZE: "+ str(len(dataset)) +" CLASS RATIO: " + str(relationRatio) +" ... BALANCING DATA")

        shuffled = shuffle(dataset)

        orderedDataset = shuffled.sort_values(by=['label'], ascending=False)
        cutOff = int((1/RELATION_RATIO)*numberOfRelations)

        balanced = shuffle(orderedDataset.head(cutOff))

        print("-----BALANCED DATASET WITH SIZE: "+str(len(balanced)))
        return balanced

    else:

        print("-----DATASET IS ALREADY BALANCED - CLASS RATIO: " + str(relationRatio)+"-----")

        return dataset

def split_into_train_and_test_set(dataset):

    DATASET_SPLIT_RATIO = 0.1

    y_data = dataset.as_matrix(columns=['label'])
    x_data = dataset.drop(['label', 'argumentationID'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=DATASET_SPLIT_RATIO, random_state=42, stratify=y_data)

    return x_train, x_test, y_train, y_test

def supervised_split(dataset, splitRatio):

    argumentationArray = np.array(list(set(dataset['argumentationID'].tolist())))
    numberOfArgumentation = argumentationArray.shape[0]
    split = int(splitRatio*numberOfArgumentation)
    testArgumentation = np.random.choice(argumentationArray, split, replace=False)

    datasetTest = dataset[dataset['argumentationID'].isin(testArgumentation)]
    datasetTrain = dataset[~dataset['argumentationID'].isin(testArgumentation)]

    print("-----DATASET SPLIT INTO TRAIN AND TEST SET - RATIO: "+str(len(datasetTrain))+" to " + str(len(datasetTest)) + "-----")

    return datasetTrain, datasetTest

def flatten_vectors(x_data):

    x_vector1 = np.stack(x_data.as_matrix(columns=['vector1']).ravel())
    x_vector1 = np.reshape(x_vector1, (x_vector1.shape[0], x_vector1.shape[1]*x_vector1.shape[2]))

    x_vector2 = np.stack(x_data.as_matrix(columns=['vector2']).ravel())
    x_vector2 = np.reshape(x_vector2, (x_vector2.shape[0], x_vector2.shape[1]*x_vector2.shape[2]))

    x_pos1 = np.stack(x_data.as_matrix(columns=['pos1']).ravel())
    x_pos1 = np.reshape(x_pos1, (x_pos1.shape[0], x_pos1.shape[1]*x_pos1.shape[2]))

    x_pos2 = np.stack(x_data.as_matrix(columns=['pos2']).ravel())
    x_pos2 = np.reshape(x_pos2, (x_pos2.shape[0], x_pos2.shape[1]*x_pos2.shape[2]))

    x_vectorPos1 = np.concatenate((x_vector1, x_pos1), axis=-1)
    x_vectorPos2 = np.concatenate((x_vector2, x_pos2), axis=-1)

    vectorFeatures = np.concatenate((x_vectorPos1, x_vectorPos2), axis=-1)

    vectorNames = ["vector1", "vector2", "pos1", "pos2"]
    columnNames = list(x_data.columns.values)
    restFeatures = [feature for feature in columnNames if feature not in vectorNames]

    otherFeatures = x_data.as_matrix(columns=restFeatures)

    x_data_output = np.concatenate((vectorFeatures,otherFeatures), axis=-1)

    return x_data_output

def prepare_data_for_NN(dataset):

    y_data = dataset.as_matrix(columns=['label'])
    numberOfLabels = np.unique(y_data).shape[0]
    y_data = np.identity(numberOfLabels)[y_data.astype(int).flatten()]

    x_data = dataset.drop(['label', 'argumentationID'], axis=1)
    x_data = flatten_vectors(x_data)

    return x_data, y_data

def prepare_data_for_RNN(dataset):

    y_data = dataset.as_matrix(columns=['label'])
    numberOfLabels = np.unique(y_data).shape[0]
    y_data = np.identity(numberOfLabels)[y_data.astype(int).flatten()]

    x_dataFrame = dataset.drop(['label', 'argumentationID'], axis=1)

    sharedFeatures = x_dataFrame.as_matrix(columns=['claimIndicatorArg1', 'premiseIndicatorArg1','claimIndicatorArg2', 'premiseIndicatorArg2','sameSentence', 'sharedNouns', 'numberOfSharedNouns', 'tokensArg1', 'tokensArg2'])

    sentence1Vector = np.stack(x_dataFrame.as_matrix(columns=['vector1']).ravel())
    sentence2Vector = np.stack(x_dataFrame.as_matrix(columns=['vector2']).ravel())
    sentence1Pos = np.stack(x_dataFrame.as_matrix(columns=['pos1']).ravel())
    sentence2Pos = np.stack(x_dataFrame.as_matrix(columns=['pos2']).ravel())

    sentence1 = np.concatenate((sentence1Vector, sentence1Pos), axis=-1)
    sentence2 = np.concatenate((sentence2Vector, sentence2Pos), axis=-1)

    x_data = {}
    x_data["sentence1"] = sentence1
    x_data["sentence2"] = sentence2
    x_data["sharedFeatures"] = sharedFeatures

    return x_data, y_data

def prepare_data_for_RF(dataset):

    y_data = dataset.as_matrix(columns=['label'])
    numberOfLabels = np.unique(y_data).shape[0]
    y_data = np.identity(numberOfLabels)[y_data.astype(int).flatten()]

    x_data = dataset.drop(['label', 'argumentationID'], axis=1)
    x_data = flatten_vectors(x_data)

    return x_data, y_data

def prepare_data_for_ZeroR(dataset):

    y_data = dataset.as_matrix(columns=['label'])
    numberOfLabels = np.unique(y_data).shape[0]
    y_data = np.identity(numberOfLabels)[y_data.astype(int).flatten()]

    x_data = dataset.drop(['label', 'argumentationID'], axis=1)
    x_data = flatten_vectors(x_data)

    return x_data, y_data
