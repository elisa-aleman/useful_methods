#-*- coding: utf-8 -*-

import os
import sys
import codecs
import csv
import scipy
import numpy
import gensim
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
import random
import sqlite3

######################################
###### Special printing methods ######
######################################
#Go up a line in the terminal to print over something
def up(): 
    # My terminal breaks if we don't flush after the escape-code
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()

#Go down a line in the terminal (like print(""))
def down(): 
    # I could use '\x1b[1B' here, but newline is faster and easier
    sys.stdout.write('\n')
    sys.stdout.flush()

#Print to terminal and to file at the same time
def printSTDlog(strlog, log_file): 
    with codecs.open(log_file, 'a', 'utf-8') as logf:
            print(strlog)
            strlog+= "\n"
            logf.write(strlog)

#Print to a file
def printLog(strlog, log_file):
    with codecs.open(log_file, 'a', 'utf-8') as logf:
            strlog+= "\n"
            logf.write(strlog) 

######################################
###### Data organizing methods #######
######################################

# Read a CSV file
def readCSV(filename, titlesplit=True):
    f = open(filename, 'rt', encoding='utf-8')
    reader = csv.reader(f, delimiter = ',')
    table = []
    for row in reader:
        table.append(tuple(row))
    f.close()
    if titlesplit:
        titles = table[0]
        table = table[1:]
        return titles, table
    else:
        return table

#Get a word list text file (a word per line) into a list
def ReadDict(filename):
    dictionary = []
    with open(filename, 'r') as thefile:
        for line in thefile:
            if (line!=""):
                nline = line.replace('\n','')
                dictionary.append(nline)
    return dictionary

#Make a list like [1,[2,3],[4,[5]]] into [1,2,3,4,5]
def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def OneHot(Y):
    uniqueY = numpy.unique(Y)
    oneHotY = numpy.zeros([Y.shape[0], uniqueY.shape[0]])
    for num, i in enumerate(Y):
        oneHotY[num][i] = 1
    return oneHotY

# get X, Y, test_x, test_y
def ReadyData(data, test_size = 1000, do_shuffle=True):
    if do_shuffle:
        numpy.random.shuffle(data)
    train_data = data[:-test_size]
    test_data = data[-test_size:]
    X,Y = zip(*train_data)
    test_x, test_y = zip(*test_data)
    X = numpy.array(list(X))#,dtype=object
    Y = numpy.array(list(Y)).reshape(-1,1)
    test_x = numpy.array(list(test_x))#,dtype=object
    test_y = numpy.array(list(test_y)).reshape(-1,1)
    return X,Y,test_x,test_y

#####################################
####### Tensorflow methods ##########
#####################################

### To check the tensorboard log
def print_log_instructions():
    print("To be able to see Tensorboard in your local machine after training on a server")
    print("    1. exit current server session")
    print("    2. connect again with the following command:")
    print("        ssh -L 16006:127.0.0.1:6006 -p [port] [user]@[server]")
    print("    3. execute in terminal")
    print("        tensorboard --logdir='{}'".format(MakeLogFile('', server=True)))
    print("    4. on local machine, open browser on:")
    print("        http://127.0.0.1:16006")

# Test NN training output
def getCurrentAverageError(model,test_x,test_y):
    pred_y = [p[0] for p in model.predict(test_x)]
    losses = [(i[0]-i[1])**2 for i in zip(pred_y,test_y)]
    mean_square_man = numpy.average(losses)
    av_error = mean_square_man**0.5
    return av_error

############################
########## Models ##########
############################

# corpus = vector >> using each title and its kinds of answers as different dimensions or "words"
# num_topics is the number of topics
# id2word = titles >> the column titles_answer are the "words" in our data
def LDA(vectorized, num_topics, vec_titles):
    vec_titles = [[i] for i in vec_titles]
    titles = gensim.corpora.Dictionary(vec_titles)
    vector = [[(key,int(val)) for key,val in enumerate(row) if int(val)!=0] for row in vectorized]
    lda = gensim.models.ldamodel.LdaModel(corpus=vector, num_topics=num_topics, id2word=titles)
    return lda

def HDP(vectorized, vec_titles):
    vec_titles = [[i] for i in vec_titles]
    titles = gensim.corpora.Dictionary(vec_titles)
    # vector = [[(key,int(val)) if val!=' ' else (key,0) for key,val in enumerate(row)] for row in vector]
    vector = [[(key,int(val)) for key,val in enumerate(row) if int(val)!=0] for row in vectorized]
    hdp = gensim.models.hdpmodel.HdpModel(corpus=vector, id2word=titles)
    return hdp

def tSNE(input_filename, output_filename, header=True, n_dim=2):
    if header:
        raw_data = numpy.genfromtxt(input_filename, delimiter=",", headerfilling_values=(0, 0, 0), skiprows=1)
    else:
        raw_data = numpy.genfromtxt(input_filename, delimiter=",", headerfilling_values=(0, 0, 0))
    compressed_data = sklearn.manifold.TSNE(n_dim).fit_transform(raw_data)
    numpy.savetxt(output_filename, compressed_data, delimiter=",")

#########################################
############## Vectorize ################
#########################################

def Vectorize(sentences, dictionary):
    # sentences: ["text",1 or 0] 1: positive, 0: negative
    vectorizer = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b')
    IM = vectorizer.fit_transform(dictionary)
    # Method 1
    X_list = []
    y_list = []
    for i in sentences:
        vector = vectorizer.transform([i[0]]).toarray().tolist()
        X_list.append(vector[0])
        y_list.append(i[1])
    X = numpy.array(X_list)
    y = numpy.array(y_list)
    return X, y

####################
### SVM Learning ###
####################

def SVM_Kfolds(x, y, k, kernel = 'linear', C = 1.0, gamma = 0.001, times = 1):
    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    testsize = len(y)/k
    # Correct Prediction, True Positive, True Negative, Incorrect Prediction, False Positive, False Negative
    counts = [{"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0} for t in xrange(k*times)]
    if type(x) == type(numpy.array([])):
        x = x.tolist()
    if type(y) == type(numpy.array([])):
        y = y.tolist()
    xysets = [row+[y[num]] for num,row in enumerate(x)]
    for t in xrange(k*times):
        numpy.random.shuffle(xysets)
        y_list = [xyset[-1] for xyset in xysets]
        X_list = [xyset[:-1] for xyset in xysets]
        X = numpy.array(X_list)
        y = numpy.array(y_list)
        #Define classifier
        clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)
        clf.fit(X[:-testsize],y[:-testsize])
        #Test data
        for i in range(1, testsize+1):
            predicted = clf.predict(X[-i].reshape(1,-1))[0]
            true_value = y[-i]
            if (predicted == true_value): # test data
                counts[t]["CP"] += 1
                if predicted == 1:
                    counts[t]["TP"] += 1
                else:
                    counts[t]["TN"] += 1
            else:
                counts[t]["IP"] += 1
                if predicted == 1:
                    counts[t]["FP"] += 1
                else:
                    counts[t]["FN"] += 1
        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        if counts[t]["TP"]+counts[t]["FP"]>0:
            precision = counts[t]["TP"] / ((counts[t]["TP"] + counts[t]["FP"])*1.0)
        else:
            precision = 0
        if counts[t]["TP"]+counts[t]["FN"]>0:
            recall = counts[t]["TP"] / ((counts[t]["TP"] + counts[t]["FN"])*1.0)
        else:
            recall = 0
        accuracy = counts[t]["CP"]/(testsize*1.0)
        if precision>0 or recall>0:
            F1 = 2* ((precision*recall)/(precision+recall))
        else:
            F1 = 0
        #
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(F1)
    avpr = sum(precisions)/(len(precisions)*1.0)
    stpr = scipy.std(precisions)
    avre = sum(recalls)/(len(recalls)*1.0)
    stre = scipy.std(recalls)
    avac = sum(accuracies)/(len(accuracies)*1.0)
    stac = scipy.std(accuracies)
    avf1 = sum(f1s)/(len(f1s)*1.0)
    stf1 = scipy.std(f1s)
    results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
    return results

def SVM_weights(x, y, feature_names, kernel = 'linear', C = 1.0, gamma = 0.001):
    if type(x) == type([]):
        x = numpy.array(x)
    if type(y) == type([]):
        y = numpy.array(y)
    clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)
    clf.fit(x,y)
    weights = clf.coef_.tolist()[0]
    influences = zip(feature_names, weights)
    return influences

def GBM_Kfolds(x, y, k, n_estimators=100, subsample=0.8, max_depth=3, times=1):
    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    testsize = len(sentences)/k
    counts = [{"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0} for t in xrange(k*times)]
    if type(x) == type(numpy.array([])):
        x = x.tolist()
    if type(y) == type(numpy.array([])):
        y = y.tolist()
    xysets = [row+[y[num]] for num,row in enumerate(x)]
    for t in xrange(k*times):
        numpy.random.shuffle(xysets)
        y_list = [xyset[-1] for xyset in xysets]
        X_list = [xyset[:-1] for xyset in xysets]
        X = numpy.array(X_list)
        y = numpy.array(y_list)
        #Define classifier
        clf = GradientBoostingClassifier(n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        clf.fit(X[:-testsize],y[:-testsize])
        #Test data
        for i in range(1, testsize+1):
            predicted = clf.predict(X[-i].reshape(1,-1))[0]
            true_value = y[-i]
            if (predicted == true_value): # test data
                counts[t]["CP"] += 1
                if predicted == 1:
                    counts[t]["TP"] += 1
                else:
                    counts[t]["TN"] += 1
            else:
                counts[t]["IP"] += 1
                if predicted == 1:
                    counts[t]["FP"] += 1
                else:
                    counts[t]["FN"] += 1
        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        if counts[t]["TP"]+counts[t]["FP"]>0:
            precision = counts[t]["TP"] / ((counts[t]["TP"] + counts[t]["FP"])*1.0)
        else:
            precision = 0
        if counts[t]["TP"]+counts[t]["FN"]>0:
            recall = counts[t]["TP"] / ((counts[t]["TP"] + counts[t]["FN"])*1.0)
        else:
            recall = 0
        accuracy = counts[t]["CP"]/(testsize*1.0)
        F1 = 2* ((precision*recall)/(precision+recall))
        #
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(F1)
    avpr = sum(precisions)/(len(precisions)*1.0)
    stpr = scipy.std(precisions)
    avre = sum(recalls)/(len(recalls)*1.0)
    stre = scipy.std(recalls)
    avac = sum(accuracies)/(len(accuracies)*1.0)
    stac = scipy.std(accuracies)
    avf1 = sum(f1s)/(len(f1s)*1.0)
    stf1 = scipy.std(f1s)
    results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
    return results


# http://xgboost.readthedocs.io/en/latest/parameter.html
def XGBoost_Kfolds(x, y, k, probability_cutoff=0.5, max_depth=2, eta=1, silent=1, objective='binary:logistic', num_round=2):
    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    testsize = len(sentences)/k
    counts = [{"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0} for t in xrange(k*times)]
    if type(x) == type(numpy.array([])):
        x = x.tolist()
    if type(y) == type(numpy.array([])):
        y = y.tolist()
    xysets = [row+[y[num]] for num,row in enumerate(x)]
    for t in xrange(k*times):
        numpy.random.shuffle(xysets)
        y_list = [xyset[-1] for xyset in xysets]
        X_list = [xyset[:-1] for xyset in xysets]
        X = numpy.array(X_list)
        y = numpy.array(y_list)
        #Define classifier
        # specify parameters via map
        param = {'max_depth':max_depth, 'eta':eta, 'silent':silent, 'objective':objective}
        dtrain = xgboost.DMatrix(X[:-testsize], label=y[:-testsize])
        clf = xgboost.train(param, dtrain, num_round)
        #Test data
        dtest = xgb.DMatrix(X[-testsize:])
        predicted_probs = clf.predict(dtest)
        ypreds = []
        for ypred in predicted_probs:
            if ypred>probability_cutoff:
                ypreds.append(1)
            elif ypred==probability_cutoff:
                ypreds.append(random.randint(0,1))
            else:
                ypreds.append(0)
        for i in range(1, testsize+1):
            predicted = ypreds[i-1]
            true_value = y[-i]
            if (predicted == true_value): # test data
                counts[t]["CP"] += 1
                if predicted == 1:
                    counts[t]["TP"] += 1
                else:
                    counts[t]["TN"] += 1
            else:
                counts[t]["IP"] += 1
                if predicted == 1:
                    counts[t]["FP"] += 1
                else:
                    counts[t]["FN"] += 1
        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        if counts[t]["TP"]+counts[t]["FP"]>0:
            precision = counts[t]["TP"] / ((counts[t]["TP"] + counts[t]["FP"])*1.0)
        else:
            precision = 0
        if counts[t]["TP"]+counts[t]["FN"]>0:
            recall = counts[t]["TP"] / ((counts[t]["TP"] + counts[t]["FN"])*1.0)
        else:
            recall = 0
        accuracy = counts[t]["CP"]/(testsize*1.0)
        F1 = 2* ((precision*recall)/(precision+recall))
        #
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(F1)
    avpr = sum(precisions)/(len(precisions)*1.0)
    stpr = scipy.std(precisions)
    avre = sum(recalls)/(len(recalls)*1.0)
    stre = scipy.std(recalls)
    avac = sum(accuracies)/(len(accuracies)*1.0)
    stac = scipy.std(accuracies)
    avf1 = sum(f1s)/(len(f1s)*1.0)
    stf1 = scipy.std(f1s)
    results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
    return results

##########################################
############## CSV and SQL ###############
##########################################

def Connect(dbname):
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    return conn,c

def CSVcreateSQL(titles, dbname, tablename):
    conn, c = Connect(dbname)
    size = len(titles)
    # titles_quotes = ["'{}'".format(i) for i in titles]
    # titles_str = ", ".join(titles_quotes)
    c.execute("CREATE TABLE {tn} ('{nf}' INTEGER PRIMARY KEY)".format(tn=tablename, nf=titles[0]))
    for title in titles[1:]:
        c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' INTEGER".format(tn=tablename, cn=title))
        conn.commit()
    conn.commit()

def CSVtoSQL(titles, table, dbname, tablename='users'):
    times = len(titles)/100
    conn, c = Connect(dbname)
    columns = titles
    columns_quote = ["'{}'".format(i) for i in columns]
    columns_str = ", ".join(columns_quote)
    sentence = "?{}".format(",?"*(len(columns)-1))
    sql = "INSERT INTO {} ({}) VALUES({})".format(tablename, columns_str, sentence)
    ins = [tuple(row) for row in table]
    c.executemany(sql,ins)
    conn.commit()
    conn.close()

def SQLtoCSV(dbname):
    # dbname = 'ctrip_db2.sqlite'
    conn, c = Connect(MakeSQLpath(dbname))
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = c.fetchall()
    tables = [table[0] for table in tables]
    for table in tables:
        sql = u"SELECT * FROM {}".format(table)
        c.execute(sql)
        raw = c.fetchall()
        columns = [des[0] for des in c.description]
        log_file = MakeSQLpath(os.path.join('CSV',"{}.csv".format(table)))
        strlog = u",".join(columns)
        printLog(strlog, log_file)
        for row in raw:
            strlog = u",".join([u'"{}"'.format(item) if type(item) == type(u'') else u'{}'.format(item) for item in row])
            printLog(strlog, log_file)

if __name__ == '__main__':
    pass
