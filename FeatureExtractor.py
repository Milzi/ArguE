import numpy as np
import logging
import requests
import pandas as pd
import gensim
import re
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
import nltk
from nltk.data import load
from keras.preprocessing.sequence import pad_sequences
import timeit
import spacy

class AdvancedFeatureExtractor:

    def __init__(self):

        self.GRAPHENE_SERVICE = "http://nietzsche.fim.uni-passau.de:8080/simplification/text"
        self.premiseIndicators = self.read_key_words("resources/premise_indicator.txt")
        self.claimIndicators = self.read_key_words("resources/claim_indicator.txt")
        self.tagdict = load('help/tagsets/upenn_tagset.pickle')
        self.lb = preprocessing.LabelBinarizer()
        self.lb.fit(list(self.tagdict.keys()))
        self.nlp=spacy.load('en')
        self.word2VecModel = gensim.models.KeyedVectors.load_word2vec_format('resources/GoogleNews-vectors-negative300.bin.gz', binary=True)

    def extractFeatures(self, dataset):

        print("-----START FEATURE EXTRACTION-----")

        extraction_time = timeit.default_timer()

        dataset = self.startFeatureExtraction(dataset)

        del dataset["arg1"]
        del dataset["arg2"]
        del dataset["originalArg1"]
        del dataset["originalArg2"]

        done = timeit.default_timer() - extraction_time
        print("Total Extraction Time:")
        print(done)

        print("-----FEATURE EXTRACTION FINISHED SIZE: "+ str(len(dataset)) + " x " + str(len(dataset.columns))+"-----")

        return(dataset)

    def startFeatureExtraction(self, dataset):

        propositionSet = list(set(dataset['arg1']))
        parsedPropositions = list()

        for proposition in propositionSet:
            words = word_tokenize(proposition)
            parsedPropositions.append(nltk.pos_tag(words))

        print("-----1. Feature: EXTRACTING WORD VECTORS-----")
        start_time = timeit.default_timer()
        dataset = self.word_vector_feature(dataset, propositionSet, parsedPropositions)
        elapsed = timeit.default_timer() - start_time
        print(elapsed)

        print("-----2. FEATURE: ADDING ONE-HOT ENCODED POS-----")
        start_time = timeit.default_timer()
        dataset = self.pos_feature(dataset, propositionSet, parsedPropositions)
        elapsed = timeit.default_timer() - start_time
        print(elapsed)

        print("-----3. Feature: ADDING KEYWORD FEATURE-----")
        start_time = timeit.default_timer()
        dataset = self.keyword_feature(dataset, propositionSet)
        elapsed = timeit.default_timer() - start_time
        print(elapsed)

        print("-----4. FEATURE: ADDING NUMBER OF TOKENS-----")
        start_time = timeit.default_timer()
        dataset = self.token_feature(dataset, propositionSet, parsedPropositions)
        elapsed = timeit.default_timer() - start_time
        print(elapsed)

        print("-----5. FEATURE: ADD SHARED NOUNS - BINARY AND VALUE----")
        start_time = timeit.default_timer()
        dataset = self.shared_noun_feature(dataset, propositionSet, parsedPropositions)
        elapsed = timeit.default_timer() - start_time
        print(elapsed)

        print("-----6. FEATURE: SAME SENTENCE FEATURE----")
        start_time = timeit.default_timer()
        dataset = self.same_sentence_feature(dataset)
        elapsed = timeit.default_timer() - start_time
        print(elapsed)

        return dataset

    def same_sentence_feature(self, dataset):

        dataset["sameSentence"] = dataset[['originalArg1','arg2']].apply(lambda row: int(bool(row['arg2'] in row['originalArg1'])), axis=1)

        return dataset

    def word_vector_feature(self, dataset, propositionSet, parsedPropositions):

        wordVectorFeature = list()
        feature_vector = np.zeros(300)

        for proposition in parsedPropositions:
            propositionVector = list()
            for word in proposition:
                if word[0] in self.word2VecModel.vocab:
                    feature_vector = self.word2VecModel[word[0]]

                propositionVector.append(feature_vector)

            wordVectorFeature.append(propositionVector)

        wordVectorFeature = np.array(wordVectorFeature)
        wordVectorFeature = pad_sequences(wordVectorFeature, value=0, padding='post', dtype=float)

        wordVectorFrame = pd.DataFrame({"arg1": propositionSet, "vector1": wordVectorFeature.tolist()})
        dataset = pd.merge(dataset, wordVectorFrame, on='arg1')
        wordVectorFrame = wordVectorFrame.rename(columns={'arg1':'arg2', "vector1":"vector2"})
        dataset = pd.merge(dataset, wordVectorFrame, on='arg2')

        return dataset


    def pos_feature(self, dataset, propositionSet, parsedPropositions):

        propositionPOSList = list()

        current = 0
        for proposition in parsedPropositions:

            propositionPOS = self.get_one_hot_pos(proposition)
            propositionPOSList.append(propositionPOS)

        propositionPOSPadded = pad_sequences(propositionPOSList, value=0, padding='post')

        posFrame = pd.DataFrame({"arg1":propositionSet, "pos1": propositionPOSPadded.tolist()})
        dataset = pd.merge(dataset, posFrame, on='arg1')
        posFrame = posFrame.rename(columns={'arg1':'arg2', "pos1":"pos2"})
        dataset = pd.merge(dataset, posFrame, on='arg2')

        return dataset

    def shared_noun_feature(self, dataset, propositionSet, parsedPropositions):

        dataset[["sharedNouns", "numberOfSharedNouns"]] = dataset[['arg1','arg2']].apply(lambda row: self.find_shared_nouns(parsedPropositions[propositionSet.index(row['arg1'])], parsedPropositions[propositionSet.index(row['arg2'])]), axis=1)

        return dataset

    def find_shared_nouns(self, proposition, partner):

        arg1Nouns = [word for (word, pos) in proposition if pos == 'NN']
        arg2Nouns = [word for (word, pos) in partner if pos == 'NN']
        intersection = set(arg1Nouns).intersection(arg2Nouns)
        shared = 0
        if len(intersection)>0:
            shared = 1
        return [shared, len(intersection)]

    def token_feature(self, dataset, propositionSet, parsedPropositions):

        numberOfTokens = list()

        for i in range(len(propositionSet)):

            numberOfTokens.append([propositionSet[i], len((parsedPropositions[i]))])

        tokenDataFrame = pd.DataFrame(data=numberOfTokens, columns=["proposition", "tokens"])

        tokenDataFrame = tokenDataFrame.rename(columns={'proposition':'arg1', 'tokens':'tokensArg1'})

        dataset = pd.merge(dataset, tokenDataFrame, on='arg1')

        tokenDataFrame = tokenDataFrame.rename(columns={"arg1" : "arg2", "tokensArg1":"tokensArg2"})

        dataset = pd.merge(dataset, tokenDataFrame, on='arg2')

        return dataset

    def padCorpus(self, corpus_tokens, wordDict):

        print("start padding")

        corpusAsWordIndex = list()
        lb =  preprocessing.LabelEncoder()
        lb.fit(wordDict)

        for sentence in corpus_tokens:
            wordIndexVector = lb.transform(sentence)
            corpusAsWordIndex.append(wordIndexVector)

        print("finished indexisation")

        corpusAsWordIndex = pad_sequences(corpusAsWordIndex, value=0, padding='post')

        print("finished padding")
        print("start reverse indexes")

        paddedCorpus = list()
        for sentence in corpusAsWordIndex:
            paddedCorpus.append(list(lb.inverse_transform(sentence)))

        return paddedCorpus

    def get_one_hot_pos(self, parsedProposition):

        posVectorList = self.lb.transform([word[1] for word in parsedProposition])
        posVector = np.array(posVectorList)

        return(posVector)

    def propositions_tokenize(self, text, keywords=False, coreference=False):

        print("Start simplification")

        url = self.GRAPHENE_SERVICE
        #change it later to this the environement variable
        #url = os.environ.get("GRAPHENE_URL")
        data = {}
        data['text'] = text
        data['doCoreference'] = False
        req = requests.post(url, json=data)
        logging.info('TEXT_SIMPLIFIED')

        # you can send more than one sentences to the service, try it later
        # print(json.dumps(req.json(),indent=4))
        for j in req.json()["simplifiedSentences"]:
            original = j['originalSentence']
            for i in j["coreSentences"]:
                if keywords:
                    yield {'text': i["notSimplifiedText"], 'keywords': self.including_keywords_features(i["notSimplifiedText"],original)}
                else:
                    yield {'text': i["notSimplifiedText"] }


    def keyword_feature(self, dataset, propositionSet):

        keyWordFeatureList = list()

        for proposition in propositionSet:

            originalSentence = dataset.loc[dataset['arg1'] == proposition]['originalArg1'].iloc[0]
            keyWordFeatureList.append(self.including_keywords_features(proposition, originalSentence))

        keywordFeatureFrame = pd.DataFrame(data=keyWordFeatureList, columns=["claimIndicatorArg1", "premiseIndicatorArg1"])
        keywordFeatureFrame["arg1"] = propositionSet

        dataset = pd.merge(dataset, keywordFeatureFrame, on='arg1')

        keywordFeatureFrame = keywordFeatureFrame.rename(columns = {'arg1':'arg2', 'claimIndicatorArg1':'claimIndicatorArg2', 'premiseIndicatorArg1': 'premiseIndicatorArg2'})

        return pd.merge(dataset, keywordFeatureFrame, on='arg2')


    def including_keywords_features(self, proposition, original):

        positionInSentence = original.find(proposition)

        if positionInSentence < 1:

            claim_indicator = self.check_claim_indicators(original[:len(proposition)])
            premise_indicator = self.check_premise_indicators(original[:len(proposition)])

        else:

            wordTokensBefore = word_tokenize(original[:positionInSentence])

            if len(wordTokensBefore) > 1:

                wordsBefore = wordTokensBefore[-2] + wordTokensBefore[-1]

            else:

                wordsBefore = wordTokensBefore[-1]

            extendedSentence = "".join(wordsBefore) + " " + proposition

            claim_indicator = self.check_claim_indicators(extendedSentence)
            premise_indicator = self.check_premise_indicators(extendedSentence)

        return [claim_indicator, premise_indicator]

    def check_premise_indicators(self, sentence):
        """
        function to detect the presence of argument keywords in a sentence
        :param full sentence:
        :return: 1 if sentence contains keyword
        """

        for indicator in self.premiseIndicators:
            if re.search(r"\b" + re.escape(indicator) + r"\b", sentence):
                return 1
        return 0

    def check_claim_indicators(self, sentence):
        """
        function to detect the presence of argument keywords in a sentence
        :param full sentence:
        :return: True if sentence contains keyword
        """
        for indicator in self.claimIndicators:
            if re.search(r"\b" + re.escape(indicator) + r"\b", sentence):
                return 1
        return 0

    def read_key_words(self, file):

        return [line.rstrip('\n') for line in open(file)]
