### evaluation interfaces
from models import *
import numpy as np
import itertools
from misc import *
import multiprocessing as mp
from sklearn import cross_validation
#from sklearn.model_selection import KFold
from keras.layers.advanced_activations import LeakyReLU,ELU
from sklearn.preprocessing import OneHotEncoder
import pickle
from collections import Counter
from keras.preprocessing import sequence
from keras.datasets import imdb
print("importing wordnet")
try:
    import nltk
    nltk.data.path.append("./nltk_data")
    nltk.data.path.append("../nltk_data")
except Exception as es:
    print(es)
from nltk.corpus import wordnet as wn
import json
from keras.datasets import reuters
from keras.datasets import imdb
from sklearn import preprocessing

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        dataset = pickle.load(f)
        x_train = dataset['x_train']
        y_train = dataset['y_train']
        x_test = dataset['x_test']
        y_test = dataset['y_test']
        enc = OneHotEncoder()
        lb = preprocessing.LabelBinarizer()
        print("One hot encoding in progress..")
        y_train = lb.fit_transform(y_train)
        y_test = lb.transform(y_test)
        y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray()
        y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray()
        word_index = dataset['mappings']
        data = ([(x_train,y_train),(x_test,y_test)],word_index)
        return data

def read_json_file(path):
    with open(path) as json_file:  
        return json.load(json_file)
    
def load_npz(fname,split_percentage=0.33):

    if "reuters" in fname:
        (x_train, y_train), (x_test, y_test)  = load_data_reuters("datasets/reuters.npz")
        word_index = read_json_file("datasets/reuters_word_index.json")
        
    elif "imdb" in fname:
        (x_train, y_train), (x_test, y_test)  = load_data_imdb("datasets/imdb.npz")
        word_index = read_json_file("datasets/imdb_word_index.json")

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
        ## is wmap really wmap?
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

if __name__ == "__main__":

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--core_dataset",default="imdb")
        parser.add_argument("--simpleNN",default=False)
        parser.add_argument("--simpleRNN",default=False)
        parser.add_argument("--hybridRNN",default=False)
        parser.add_argument("--RF",default=False)
        parser.add_argument("--mergedRF",default=False)
        parser.add_argument("--svm",default=False)
        parser.add_argument("--mergedsvm",default=False)
        parser.add_argument("--hybridNN",default=False)
        parser.add_argument("--num_features",default=1000) ## number of semantic features generated
        parser.add_argument("--maxlen",default=50,type=int) ## number of words used for classification
        args = parser.parse_args()
        print(args)
        import nltk

        if ".pkl" in args.core_dataset:
                print("Loading the PKL file..")
                dataset,dmap  = load_obj(args.core_dataset)

        else:
            dataset,dmap = load_npz(args.core_dataset)

        sstrain,sstest = dataset        
        total_test = np.concatenate((sstrain[1],sstest[1]),axis=0)
        total_train = []

        ## concatenate whole dataset
        for x in sstrain[0]:
            total_train.append(x)
        for x in sstest[0]:
            total_train.append(x)
        
        #kf = KFold(n_splits=10)
        
        ## in this example, stratify according to the 0 class
        kf = cross_validation.StratifiedKFold(total_test[:,0], n_folds=10)
        current_fold = 0
        
        ## perform 10 repetitions
        for k in range(10):
            ## k - fold cross validation
            for train_index, test_index in kf:
                current_fold +=1
                print("Starting a new fold..")
                x_train = [total_train[x] for x in train_index]
                y_train = [total_train[x] for x in test_index]
                x_test = total_test[train_index]
                y_test = total_test[test_index]            
                dataset = ((x_train,x_test),(y_train,y_test))          
                dataset = wordnet_features(dataset, dmap,num_features=int(args.num_features),padding=args.maxlen)

                ## to pognati v loopu cez vse mozne folde..
                print("Proceeding with the learning phase..")

                ## do not use semantic features
                if args.simpleNN:
                        predictions,test = simple_conv_architecture(dataset,maxlen=args.maxlen)
                        parse_results(predictions,test,"simpleNN",maxlen=args.maxlen,num_features=args.num_features,dataset=args.core_dataset)

                if args.simpleRNN:
                        predictions,test = simple_RNN_architecture(dataset,maxlen=args.maxlen)
                        parse_results(predictions,test,"simpleRNN",maxlen=args.maxlen,num_features=args.num_features,dataset=args.core_dataset)

                ## a specialized architecture for semantic features
                if args.hybridRNN:
                        predictions,test = hybrid_rnn_architecture(dataset,maxlen=args.maxlen)
                        parse_results(predictions,test,"hybridRNN",maxlen=args.maxlen,num_features=args.num_features,dataset=args.core_dataset)

                if args.RF:
                    print("testing RFs..")
                    predictions,test = baseline_rf(dataset,maxlen=args.maxlen)
                    parse_results(predictions,test,"RF",maxlen=args.maxlen,num_features=args.num_features,dataset=args.core_dataset)

                if args.mergedRF:
                    print("testing semantic RFs..")
                    predictions,test = baseline_rf(dataset,maxlen=args.maxlen,semantic=True)
                    parse_results(predictions,test,"mergedRF",maxlen=args.maxlen,num_features=args.num_features,dataset=args.core_dataset)

                if args.svm:
                    print("testing SVMs..")
                    predictions,test = baseline_svm(dataset,maxlen=args.maxlen,semantic=False)
                    parse_results(predictions,test,"SVM",maxlen=args.maxlen,num_features=args.num_features,dataset=args.core_dataset)

                if args.mergedsvm:
                    print("testing semantic SVMs..")
                    predictions,test = baseline_svm(dataset,maxlen=args.maxlen,semantic=True)
                    parse_results(predictions,test,"mergedSVM",maxlen=args.maxlen,num_features=args.num_features,dataset=args.core_dataset)
