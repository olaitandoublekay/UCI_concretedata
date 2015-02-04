"""
************************************************************************************************************************
Code by Shiv Surya
Splits Concrete Compressive Strength Dataset into a training,validation and testing set
Details of the Dataset:http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength

|Data Set Characteristics  : Multivariate || Number of Instances  :1030 || Area               : Physical   |
|Attribute Characteristics : Real         || Number of Attributes : 9   || Date Donated       : 2007-08-03 |
|Associated Tasks          : Regression   || Missing Values       ? N/A || Number of Web Hits : 58514      |

************************************************************************************************************************
"""
#import necessary packages
import csv
import numpy as np
from sklearn.preprocessing import scale



#read the data in csv file "concretedata.csv"
reader_data=csv.reader(open("concretedata.csv","rb"),delimiter=',');

#skip header
next(reader_data,None)

#save data as a numpy array
x=list(reader_data)
data=np.array(x).astype('float')

no_train=656
no_val=219
noShuffle=10

#shuffle data row-wise "noShuffle" times
#for i in range(0,noShuffle):
#    np.random.shuffle(data)

#scale data
#data=scale(data,axis=0,with_mean=True,with_std=False)

#split database as 600 samples in training set, 200 in validation set and 230 samples in testing set
X_train=data[0:no_train,0:data.shape[1]-2]
Y_train=data[0:no_train,data.shape[1]-1]

X_val= data[no_train:no_train+no_val,0:data.shape[1]-2]
Y_val= data[no_train:no_train+no_val,data.shape[1]-1]

X_test= data[no_train+no_val:data.shape[0],0:data.shape[1]-2]
Y_test= data[no_train+no_val:data.shape[0],data.shape[1]-1]

#save features
np.savetxt("X_train.csv",X_train,delimiter=",");
np.savetxt("X_test.csv",X_test,delimiter=",");
np.savetxt("X_val.csv",X_val,delimiter=",");

#save labels
np.savetxt("Y_train.csv",Y_train,delimiter=",");
np.savetxt("Y_test.csv",Y_test,delimiter=",");
np.savetxt("Y_val.csv",Y_val,delimiter=",");


print len(Y_val)
print len(Y_test)
print len(Y_train)