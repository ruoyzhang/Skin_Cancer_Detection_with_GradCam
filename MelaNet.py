from torch.utils.data.dataset import Dataset
import cv2
import pandas as pd
import numpy as np
import os


class MelaData(Dataset):
    def __init__(self, data_dir, label_csv, transforms=None):
        """
        data_dir : path to images folder
        label_csv : path to HAM10000_metadata.csv
        """
        files = os.listdir(data_dir)
        self.len = len(files)
        
        images = cv2.imread(data_dir + files[0])
        images = np.expand_dims(images, axis = 0)
        for i in range(1, len(files)):
            im = cv2.imread(data_dir + files[i])
            im = np.expand_dims(im, axis = 0)
            images = np.concatenate((images, im), axis = 0)
        self.images = images
        
        dx_to_num = {'nv' : 0, 'mel': 1, 'bkl': 2, 'df': 3, 'akiec': 4, 'bcc': 5, 'vasc' : 6}        
        labels = pd.read_csv(label_csv)
        labels['image_id_num'] = labels['image_id'].apply(lambda x: int(x.strip('ISIC_')))
        labels = labels.sort_values(by = 'image_id_num')        
        labels.reset_index(inplace = True, drop = True)
        labels['label'] = labels['dx'].apply(lambda x: dx_to_num[x])
        
        self.labels = labels
        
        self.transforms = transforms
        
    def __getitem__(self, index):
        img = self.images[index, :, :, :]
        img_t = torch.from_numpy(img) 
        label = self.labels.loc[index].label
        label = np.array(label)
        label_t = torch.from_numpy(label)
        return (img_t, label_t)

    def __len__(self):
        return self.len


def train_val_test_split(dataset, train_split, val_split, test_split):
	"""
	Split data set into training, validation, and test sets.
	"""
    if train_split + val_split + test_split != 1:
        print('Incorrect split sizes')
    
    # Size of data set
    N = dataset.__len__()
    
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
    train_data = torch.utils.data.Subset(dataset, train_ids)
    val_data = torch.utils.data.Subset(dataset, val_ids)
    test_data = torch.utils.data.Subset(dataset, test_ids)
    
    return train_data, val_data, test_data