#written by Shiv Surya
#Reads features and labels written by dataSplit.py in csv format

import csv
import numpy as np

def readData():
    #read features
    reader_data=csv.reader(open("X_train1.csv","rb"),delimiter=',')
    x=list(reader_data)
    X_train=np.array(x).astype('float')

    reader_data=csv.reader(open("X_val1.csv","rb"),delimiter=',')
    x=list(reader_data)
    X_val=np.array(x).astype('float')

    reader_data=csv.reader(open("X_test1.csv","rb"),delimiter=',')
    x=list(reader_data)
    X_test=np.array(x).astype('float')

    #read labels
    reader_data=csv.reader(open("Y_train1.csv","rb"),delimiter=',')
    x=list(reader_data)
    Y_train=np.array(x).astype('float')

    reader_data=csv.reader(open("Y_val1.csv","rb"),delimiter=',')
    x=list(reader_data)
    Y_val=np.array(x).astype('float')

    reader_data=csv.reader(open("Y_test1.csv","rb"),delimiter=',')
    x=list(reader_data)
    Y_test=np.array(x).astype('float')

    return X_train,Y_train,X_val,Y_val,X_test,Y_test


