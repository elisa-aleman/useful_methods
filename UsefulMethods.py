#-*- coding: utf-8 -*-

import os
import sys
import codecs
import csv
import scipy
import numpy
from sklearn.model_selection import train_test_split
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
    x, y = zip(*data)
    X, test_x, Y, test_y = train_test_split(x,y, test_size=test_size, shuffle=do_shuffle)
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
