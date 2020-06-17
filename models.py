import numpy as np
#from nltk.wsd import lesk
import itertools
import multiprocessing as mp
from keras.layers.advanced_activations import LeakyReLU,ELU
from sklearn.preprocessing import OneHotEncoder
import pickle
from collections import Counter
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.callbacks import *
from keras.layers import Dense, Dropout, Activation, Input, Embedding,Bidirectional, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D,Concatenate,LSTM,TimeDistributed,MaxPooling1D,Reshape
from keras.models import Model
from keras.datasets import imdb
from nltk.corpus import wordnet as wn
import json
from keras.datasets import reuters
from keras.datasets import imdb
from sklearn import preprocessing
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=0.2, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def return_best_model(compiled_model,x_train,labels_train,tolerance_threshold=1,epochs=4):

    from sklearn.model_selection import StratifiedShuffleSplit
    history = []
    tmp_optimum = 0
    tolerance = 0
    best_weights = None

    ## split train into train and validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=40)
    train_index,test_index = next(sss.split(x_train,labels_train))

    XTRAIN = x_train[train_index]
    XLABELS = labels_train[train_index]

    YTRAIN = x_train[test_index]
    YLABELS = labels_train[test_index]

    for x in range(epochs):            
        history = compiled_model.fit(XTRAIN, XLABELS,
                               batch_size=48,
                               epochs=1,
                               validation_data=(YTRAIN, YLABELS))
        scores = compiled_model.evaluate(YTRAIN, YLABELS, verbose=0)
        if scores[1] > tmp_optimum:
            print("Improving the model.. {}".format(scores[1]))
            tmp_optimum = scores[1]
            best_weights = compiled_model.get_weights()
        else:
            tolerance += 1
            if tolerance == tolerance_threshold:
                break
            
    compiled_model.set_weights(best_weights)
    return compiled_model

def load_npz(fname,split_percentage=0.33):

    if "reuters" in fname:
        (x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                                 num_words=None,
                                                                 skip_top=0,
                                                                 maxlen=None,
                                                                 test_split=split_percentage,
                                                                 seed=113,
                                                                 start_char=1,
                                                                 oov_char=2,
                                                                 index_from=3)
        word_index = reuters.get_word_index(path="reuters_word_index.json")
        
    elif "imdb" in fname:
        (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                              num_words=None,
                                                              skip_top=0,
                                                              maxlen=None,
                                                              seed=113,
                                                              start_char=1,
                                                              oov_char=2,
                                                              index_from=3)

        word_index = imdb.get_word_index(path="imdb_word_index.json")

#        out_object = {"x_train" : X_train,"x_test" : X_test,"y_train":y_train,"y_test":y_test,"mappings" : reverse_mapping}

    else:
        print("Processing the PKL dataset.")
        le = preprocessing.LabelEncoder()
        data = load_obj(fname)
        x_train = np.array(data['x_train'])
        y_train = le.fit_transform(data['y_train'])
        x_test = np.array(data['x_test'])
        y_test = le.fit_transform(data['y_test'])
        print(y_test.shape,y_train.shape)
        word_index = data['mappings']

    enc = OneHotEncoder()
    print("One hot encoding in progress..")
    y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray()
    y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray()

    return ([(x_train,y_train),(x_test,y_test)],word_index)

def data_processing_kernel(vec):
    subsets = []
    for index, k in enumerate(vec):
        if k in _reversed_wmap and index <= _padding:
            a = _reversed_wmap[k]
            if a in _ldct.keys():
                subsets.append(_ldct[a])
    try:
        fvect = np.zeros(_num_features)
        semantic_space = set.union(*subsets)
        coverage = len(semantic_space.intersection(_feature_set))
        for ind, x in enumerate(_feature_set):
            if x in semantic_space:
                fvect[ind] = _term_counts[x]                
        return fvect
    except:
        return fvect ## empty -- no semantic mapping present -- excluded in feature refinement.

def vector_job_generator(data):
    for vec in data:
        yield (vec)

def generate_feature_vectors(data,reversed_wmap,ldct,feature_set,num_features,term_counts,padding=100):

    ## not making this global introduces memory overhead.
    
    global _reversed_wmap
    global _ldct
    global _feature_set
    global _num_features
    global _term_counts
    global _padding
    
    ## generate features from ldct, which is based on a distribution of some sort
    novel_feature_vectors = []
    unannotated = 0

    for enx,vec in enumerate(data):
        if enx % 3000 == 0:
            print("Processed {}% of data.".format((enx*100)/len(data)))

        subsets = []
        for index, k in enumerate(vec):                        
            if k in reversed_wmap.keys() and index <= padding:
                a = reversed_wmap[k]
                if a in ldct.keys():
                    subsets.append(ldct[a])
        try:
            fvect = np.zeros(num_features)
            semantic_space = set.union(*subsets)
            coverage = len(semantic_space.intersection(feature_set))
            for ind, x in enumerate(feature_set):
                if x in semantic_space:
                    fvect[ind] = term_counts[x]
            novel_feature_vectors.append(fvect)
                
        except:
            unannotated+=1
            novel_feature_vectors.append(fvect)
            
    return np.matrix(novel_feature_vectors)


## to moras lociti na posamezne komponente..
def wordnet_features(data, mapJSON, num_features = 500, padding = 100 ,cutoff= 10, scale_factor=5):

        (x_train,labels_train),(x_test,labels_test) = data
        wmap = mapJSON
        reversed_wmap = {v : k for k,v in wmap.items()}
        flist = []
        tw = len(wmap.keys())
        ldct = {}
        cnts = {}
        
        processed = 0 
        for word in wmap.keys():
                processed += 1
                if processed % 15000 == 0:
                        print("Processed {} words.".format(str(processed)))
                ss = wn.synsets(word)
                total_wordset = []
                if len(ss) > 0:
                        for syn in ss:

                            ## this can be done in parallel
                            paths = syn.hypernym_paths()
                            common_terms = []
                            for path in paths:
                                for x in path:
                                    common_terms.append(x.name())
                            common_terms = set(common_terms)
                            total_wordset.append(common_terms)
                                
                if len(total_wordset) > 0:
                        total_wordset = set.union(*total_wordset)
                        ldct[word] = total_wordset
                        for x in total_wordset:
                                if x not in cnts.keys():
                                        cnts[x] = 1
                                else:
                                        cnts[x] += 1
        to_del = []
        for k,w in cnts.items():
                if w < cutoff:
                        to_del.append(k)
                        
        for j in to_del:
                del cnts[j]

        feature_set = []
        added_features = 0

        pre_features = num_features#*scale_factor
        for k in sorted(cnts, key=cnts.get,reverse=False):
                if added_features >= pre_features:
                        break
                else:
                    coin = 1 #np.random.randint(2, size=1)
                    if coin:
                        feature_set.append(k)
                        added_features += 1

        feature_set = set(feature_set)

        print("... Generating semantic vectors ...")
        novel_feature_vectors_train = generate_feature_vectors(x_train,reversed_wmap,ldct,feature_set,pre_features,cnts,padding=padding)        
        novel_feature_vectors_test = generate_feature_vectors(x_test,reversed_wmap,ldct,feature_set,pre_features,cnts,padding=padding)
        
        print("Generated novel feature vectors {} {}".format(novel_feature_vectors_train.shape,novel_feature_vectors_test.shape))
        ## return the vectors
        return [(x_train,labels_train),(x_test,labels_test),(novel_feature_vectors_train,novel_feature_vectors_test)]

def parallel_selector(threshold_integer,num_features,selected_features,lower_feature_bound):
    new_selected_features = np.sum(np.mean(selected_features, axis=0) > threshold_integer)
    count_difference = num_features - new_selected_features
    if count_difference >= 0:
        return (threshold_integer,True,count_difference)
    
    elif count_difference > lower_feature_bound:
        return (threshold_integer,False,count_difference)
    
def simple_conv_architecture(data,maxlen=500):

        ## process data
        (x_train,labels_train),(x_test,labels_test), (semantic_train,semantic_test) = data
        
        # set parameters:3
        max_features = 100000
        batch_size = 5
        embedding_dims = int(maxlen/2)
        filters = embedding_dims
        kernel_size = 5
        hidden_dims = filters
        epochs = 3

        x_train = [x[0:maxlen] for x in x_train]
        x_test = [x[0:maxlen] for x in x_test]
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

        model = Sequential()

        model.add(Embedding(max_features,
                            embedding_dims,
                            input_length=maxlen))
        
        model.add(Dropout(0.2))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))

        # we use max pooling:
        model.add(GlobalMaxPooling1D())
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(labels_train.shape[1]))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'])

        best_model = return_best_model(model,x_train,labels_train,epochs=3)
        predictions = best_model.predict(x_test)
        labels = np.zeros(predictions.shape)
        labels[predictions>0.5] = 1
        return (labels,labels_test)

def simple_RNN_architecture(data,maxlen=500):

        ## process data
        (x_train,labels_train),(x_test,labels_test), (semantic_train,semantic_test) = data

        # set parameters
        max_features = 100000
        semantic_embedding_dims = 100
        batch_size = 5
        embedding_dims = int(maxlen/2)
        filters = embedding_dims
        hidden_dims = filters
        epochs = 3

        semantic_shape_hidden = hidden_dims
        semantic_shape = semantic_train.shape[1]
        
        x_train = [x[0:maxlen] for x in x_train]
        x_test = [x[0:maxlen] for x in x_test]

        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

        ## a hybrid model with two inputs!
        input1 = Input(shape=(maxlen,))
        e1 = Embedding(max_features,embedding_dims)(input1)
        d_zero = Dropout(0.1)(e1)
        c1 = LSTM(16)(d_zero)
        d1 = Dropout(0.5)(c1)
        de1 = Dense(hidden_dims)(d1)
        d1_1 = Dropout(0.3)(de1)        
        mix1 = Dense(32)(d1_1)
        dp_1 = Dropout(0.3)(mix1)
        da_2 = ELU()(dp_1)
        mix2 = Dense(32)(da_2)
        out = Dense(labels_train.shape[1],activation="sigmoid")(mix2)
        model = Model(inputs=[input1], outputs=out)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'])
        callbacks = [
          #  EarlyStoppingByLossVal(monitor='loss', value=0.2, verbose=1)
        ]
        print(model.summary())
        
        model.fit([x_train], labels_train,
                       batch_size=24,
                       epochs=5,verbose=2)

        predictions = model.predict([x_test])
        labels = np.zeros(predictions.shape)
        labels[predictions>0.5] = 1
        return (labels,labels_test)

def hybrid_rnn_architecture(data, maxlen=500):

        ## process data
        (x_train,labels_train),(x_test,labels_test), (semantic_train,semantic_test) = data

        # set parameters
        max_features = 100000
        semantic_embedding_dims = 100
        batch_size = 5
        embedding_dims = int(maxlen/2)
        filters = embedding_dims
        kernel_size = 5
        hidden_dims = filters
        epochs = 3

        semantic_shape_hidden = hidden_dims
        semantic_shape = semantic_train.shape[1]
        
        x_train = [x[0:maxlen] for x in x_train]
        x_test = [x[0:maxlen] for x in x_test]

        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

        ## a hybrid model with two inputs!
        input1 = Input(shape=(maxlen,))
        e1 = Embedding(max_features,embedding_dims)(input1)
        d1 = Dropout(0.1)(e1)
        c1 = LSTM(32,activation='relu', return_sequences = True)(d1)
        distributed = TimeDistributed(Dense(1))(c1)
        poolc1 = MaxPooling1D()(distributed)
        de1 = Dense(hidden_dims)(poolc1)
        d1_1 = Dropout(0.25)(de1)
        input2 = Input(shape=(semantic_shape,))        
        e2_2 = Embedding(128, semantic_embedding_dims, input_length=semantic_shape)(input2)
        e2_x = Flatten()(e2_2)
        d2_0 = Dense(64)(e2_x)
        activation_1 = ELU()(d2_0)
        drop_2 = Dropout(0.3)(activation_1)
        d2_1 = Dense(hidden_dims)(drop_2)
        pm = ELU()(d2_1)
        dim1 = d1_1.shape[1] * d1_1.shape[2]
        dim2 = pm.shape[1]
        l1 = Reshape([dim1])(d1_1)
        l2 = Reshape([dim2])(pm)
        added = Concatenate()([l1,l2])
        mix1 = Dense(100)(added)
        dp_1 = Dropout(0.3)(mix1)
        da_2 = ELU()(dp_1)
        mix2 = Dense(50)(da_2)
        out = Dense(labels_train.shape[1],activation="sigmoid")(mix2)
        model = Model(inputs=[input1, input2], outputs=out)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'])
        callbacks = [
          #  EarlyStoppingByLossVal(monitor='loss', value=0.2, verbose=1)
        ]
        print(model.summary())
        
        model.fit([x_train,semantic_train], labels_train,
                       batch_size=8,
                       epochs=5)

        predictions = model.predict([x_test,semantic_test])
        labels = np.zeros(predictions.shape)
        labels[predictions>0.5] = 1
        return (labels,labels_test)


def baseline_rf(data, maxlen = 500,semantic = False, tfidf_transform = True):

#    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    from sklearn.multiclass import OneVsRestClassifier
    
    (x_train,labels_train),(x_test,labels_test), (semantic_train,semantic_test) = data
        
    x_train = [x[0:maxlen] for x in x_train]
    x_test = [x[0:maxlen] for x in x_test]

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    if tfidf_transform:
        print ("Doing the TFIDF transformation..")
        from sklearn.feature_extraction.text import TfidfTransformer
        tfx = TfidfTransformer()
        x_train = tfx.fit_transform(x_train)
        x_test = tfx.transform(x_test)
    
    if semantic:
        x_train = np.concatenate((x_train,semantic_train),axis=1)
        x_test = np.concatenate((x_test,semantic_test),axis=1)    

    ## dodaj tfidf
    clf_ovr = RandomForestClassifier(n_estimators=100,n_jobs=4)
    clf_ovr = OneVsRestClassifier(clf_ovr,n_jobs=1)

    clf_ovr.fit(x_train, labels_train)
    return (clf_ovr.predict(x_test),labels_test)


def baseline_svm(data, maxlen = 500,semantic = False, tfidf_transform = True):

#    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    from sklearn.multiclass import OneVsRestClassifier
    
    (x_train,labels_train),(x_test,labels_test), (semantic_train,semantic_test) = data
        
    x_train = [x[0:maxlen] for x in x_train]
    x_test = [x[0:maxlen] for x in x_test]

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    from sklearn.feature_extraction.text import TfidfTransformer
    rx = TfidfTransformer()
    x_train = rx.fit_transform(x_train)
    x_test = rx.transform(x_test)
    
    if semantic:
        try:
            x_train = np.concatenate((x_train,semantic_train),axis=1)
            x_test = np.concatenate((x_test,semantic_test),axis=1)
        except:
            pass

    clf_ovr = OneVsRestClassifier(SVC(kernel="rbf",C=1))
    clf_ovr.fit(x_train, labels_train)
    
    #clf_ovr.fit(x_train, labels_train)
    return (clf_ovr.predict(x_test),labels_test)

def parse_results(y_pred,y_true,method = "default",maxlen=500,num_features=1000,dataset="test"):
    from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
    if y_true.shape[1] > 2:

        ## take max from each row
        print("multiple possible classes")
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true,y_pred,average="micro")
        recall = recall_score(y_true,y_pred,average="micro")
        precision = precision_score(y_true,y_pred,average="micro")
        
    else:
        try:
            y_pred = y_pred[:,0]
        except:
            y_pred = y_pred
        y_true = y_true[:,0]
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true,y_pred)
        precision = precision_score(y_true,y_pred)
        recall = recall_score(y_true,y_pred)
        
    print("EVALUATION",str(acc),str(f1),str(precision),str(recall),method,str(maxlen),str(num_features),dataset)
