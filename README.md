# Useful Methods
Useful methods I like to import to save time

## Special printing methods

* __up()__ : Go up a line in the terminal to print over something
* __down()__ : Go down a line in the terminal
* __printSTDlog(strlog, log_file)__ : Print to terminal and to file at the same time
* __printLog(strlog, log_file)__ : Print to a file

## Data organizing methods

* __readCSV()__ : Read a CSV file into python list
* __ReadDict()__ : Get a word list text file (a word per line) into a list
* __flatten(container)__ : Make a list like [1,[2,3],[4,[5]]] into [1,2,3,4,5]
* __OneHot(Y)__ : Change a list of integers like [1,2,0] into a One-Hot encoding, [[0,1,0],[0,0,1],[1,0,0]]
* __ReadyData()__ : get X, Y, test_x, test_y from a numpy data file

## Tensorflow methods 

```
To be able to see Tensorboard in your local machine after training on a server
    1. exit current server session
    2. connect again with the following command:
        ssh -L 16006:127.0.0.1:6006 -p [port] [user]@[server]
    3. execute in terminal")
        tensorboard --logdir= [your log directory]
    4. on local machine, open browser on:
        http://127.0.0.1:16006
```
        
* __getCurrentAverageError(model,test_x,test_y)__ : Test NN training output

## Models 

* __LDA()__ : Simple gensim implementation of LDA model
* __HDP()__ : Simple gensim implementation of HDP model
* __tSNE()__ : Scikit-learn implementation of tSNE model

## Vectorize 

* __Vectorize(sentences, dictionary)__ : Vectorize a list of sentences into bag of words

## Machine Learning

* __SVM_Kfolds()__ : SVM K-fold cross validation with Accuracy, F1, Precision and Recall outputs
* __SVM_weights()__ : access a trained SVM weights
* __GBM_Kfolds()__ : K-fold cross validation with Accuracy, F1, Precision and Recall outputs
* __XGBoost_Kfolds()__ : XGBoost K-fold cross validation with Accuracy, F1, Precision and Recall outputs

## CSV and SQL 

* __CSVcreateSQL(titles, dbname, tablename)__ : create empty SQL table from a csv file column titles
* __CSVtoSQL(titles, table, dbname, tablename)__ : copy csv into the created SQL table
* __SQLtoCSV(dbname)__ : make a csv by table from a SQL database
 
