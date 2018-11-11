# ArguE
## RNN implementation for argumentative discourse relation classification


ArguE's Argumentation Classifier can be started using the interface called ArguE. 

Inside that python file you will find an interface and a main class for initiating it. Once initialised you can use ArguE's methods 
to load the data from the XML-files, extract the features and train and test the different classifiers. 
Due to the size of the corpora, it makes sense to save the data into an h5.file after each extraction step, you will find 
the appropriate helper methods inside the interface. However, these parts have already been done. 
You can find the loaded datasets including the features inside the resource/datasets directory on google drive (too big for github) (unpack the zip file first). 
The folders labelled training and testing contain datasets that are already splitted into train and test sets. 
You can find methods for loading the data from the h5.files inside the interface. An example can be found inside the main class.

It is also recommended, that you save the trained classifier models after each training process, the needed methods are provided. 
Some example models are already saved in the resources/classifierModels directory (on google drive). For the feature extraction, it is essential 
that the pre-trained google news word-vector model is inside the resource folder as well as the premise and claim indicator 
text files.

