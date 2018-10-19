import os
import pandas as pd
import math
import numpy as np
import sys

def train_val_test_split(labels,train_split = 0.7, val_split = 0.2, test_split = 0.1):
    
    # Size of data set
    N = labels.shape[0]
    
    # Size of train set
    train_size = math.floor(train_split * N)
    
    # Size of validation set
    val_size = math.floor(val_split * N)
    
    # List of all data indices
    indices = list(range(N))
    
    # Random selection of indices for train set
    train_ids = np.random.choice(indices, size=train_size, replace=False)
    train_ids = list(train_ids)
    
    # Deletion of indices used for train set
    indices = list(set(indices) - set(train_ids))
    
    # Random selection of indices for validation set
    val_ids = np.random.choice(indices, size=val_size, replace=False)
    val_ids = list(val_ids)
    
    # Selecting remaining indices for test set
    test_ids = list(set(indices) - set(val_ids))
    
    # Creating subsets
    train_data = labels.iloc[train_ids]
    val_data = labels.iloc[val_ids]
    test_data = labels.iloc[test_ids]
    
    return [train_data, val_data, test_data]

def sep_classes(df, doc_name):
    cats = list(df.dx.unique())
    for cat in cats :
    	line = ''
    	names = list(df[df.dx == cat].image_id.values)
    	names = [x  + '.jpg ' for x in names]
    	for n in names:
    		line += n
    	with open(doc_name + '_' + str(cat) + '.txt' , 'w') as f:
    		for i in range(0,len(cats)):
    			f.write(line)


if __name__ == '__main__':
	df_dir = sys.argv[1]
	labels = pd.read_csv(df_dir)
	split = train_val_test_split(labels)
	train = split[0]
	val = split[1]
	test = split[2]
	sep_classes(train, 'train')
	sep_classes(val, 'val')
	sep_classes(test, 'test')

