import torch.utils.data as data
import time
import os.path
import torch
import numpy as np
from PIL import Image
import os
import pandas as pd

from skimage import io
from skimage import transform

def default_loader(path):
    
    img = io.imread(path)
    
    return img


def get_img_list(dir, fold, label_use, csv_file, mode):
    img_list = []
    label_list = []

    img_dir = os.path.join(dir, fold + 'images/')
    csv_labels = str(dir + fold + '/' + csv_file)
    labels = pd.read_csv(csv_labels, usecols=['Image_ID', 'Labels'])

    #pegar 25% do dataset
    tam_dataset = labels.shape[0]*0.25

    if mode == 'train':
        tam_list = int(0.8*tam_dataset)
        labels = labels.iloc[0:tam_list]
    else:
        tam_list = int(0.2*tam_dataset)
        labels = labels.iloc[-tam_list:]

    if label_use == False:
    	for i in range (tam_list):
    		label_list.append('')
    		img = labels.iloc[i]['Image_ID']
    		img_list.append(img)

    else:
	    for i in range (tam_list):

	        lab = labels.iloc[i]['Labels']  
	        img = labels.iloc[i]['Image_ID']

	        try:
	            lab = lab.lower().split('|')
	        except:
	            lab = ['no finding']

	        if not all(x in lab for x in ['cardiomegaly', 'atelectasis']):
	            label_list.append(lab)
	            img_list.append(img)
	        if not all(x in lab for x in ['cardiomegaly', 'pneumonia']):
	            label_list.append(lab)
	            img_list.append(img)
	        if not all(x in lab for x in ['atelectasis', 'pneumonia']):
	            label_list.append(lab)
	            img_list.append(img)
	        if not all(x in lab for x in ['cardiomegaly', 'atelectasis', 'pneumonia']):
	            label_list.append(lab)
	            img_list.append(img)

    tam_classm1 = 1
    tam_class0 = 1
    tam_class1 = 1
    tam_class2 = 1
    tam_class3 = 1
    for j in range (len(label_list)):
        if set(label_list[j]) == '':
            label_list[j] = -1
            tam_classm1 = tam_classm1 + 1

        elif 'cardiomegaly' in label_list[j]:
            label_list[j] = 0
            tam_class0 = tam_class0 + 1
        elif 'atelectasis' in label_list[j]:
            label_list[j] = 1
            tam_class1 = tam_class1 + 1
        elif 'pneumonia' in label_list[j]:
            label_list[j] = 2
            tam_class2 = tam_class2 + 1
        else:
            label_list[j] = 3
            tam_class3 = tam_class3 + 1

    pesos = [tam_class0, tam_class1, tam_class2, tam_class3]


    return img_list, label_list, pesos

'''
list_im, list_label, pesos = get_img_list('/home/CADCOVID/Datasets_CoDAGANs/', 'dataset1/', True, 'dataset1.csv', 'train')
print(list_im)
print(list_label)
print(pesos)
'''

class ImageFolder(data.Dataset):

    def __init__(self, data_root, mode, n_datasets, label_use, resize_to=(256, 256)):

        datasets = list(range(n_datasets))
        folds = ['dataset' + str(num) for num in datasets]
        files_csv = [fold + '.csv' for fold in folds]
        label_use = [int(l) > 0 for l in label_use.split('|')]

        imgs = []
        labels = []
        pesos = []

        for i in range (n_datasets):
            img, label, peso = get_img_list(data_root, folds[i], label_use[i], files_csv[i], mode)
            imgs.append(img)
            labels.append(label)
            pesos.append(peso)

        self.data_root = data_root
        self.folds = folds
        self.mode = mode
        self.imgs = imgs
        self.labels = labels
        self.pesos = pesos
        
        self.label_use = label_use
        self.resize_to = resize_to
        

    def load_samples(self, n_samples, d_index):
        
        sample_list = [self.load_preprocess(self.imgs[d_index][i], d_index) for i in range(n_samples)]
        
        img_list = [s[0] for s in sample_list]
        label_list = [self.labels[d_index][i] for i in range(n_samples)]
        lbl_list = [s for s in label_list]
        
        return img_list, lbl_list
        
    def load_preprocess(self, img_path, d_index):
        
        # Loading.
        file_name = str(self.data_root + self.folds[d_index] + '/images/' + img_path)
        img = io.imread(file_name)
        
        # Resizing.
        img = transform.resize(img, self.resize_to, order=1, preserve_range=True)
        
        # Normalization and transformation to tensor.
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) - 0.5
        img = np.expand_dims(img, axis=0)

        if len(img.shape) != 3:
        	img = img[:,:,:,0]

        img = torch.from_numpy(img)
 
        return img


    def __getitem__(self, index):
        
        # Randomly choosing dataset.
        t = time.time()
        seed = int((t - int(t)) * 100) * index
        np.random.seed(seed)
        
        perm = np.random.permutation(len(self.imgs))[:2]

        
        ind_a = perm[0]
        ind_b = perm[1]

        #print(ind_a)
        
        # Label.
        label_a = self.labels[ind_a][index]
        label_b = self.labels[ind_b][index]
        
        # Computing paths.
        img_path_a = self.imgs[ind_a][index]
        img_path_b = self.imgs[ind_b][index]
        
        # Load function.
        img_a = self.load_preprocess(img_path_a, ind_a)
        img_b = self.load_preprocess(img_path_b, ind_b)
        
        
        return [img_a, img_b, ind_a, ind_b, label_a, label_b]

    def __len__(self):
        
        return min([len(img) for img in self.imgs])

'''
root = '/home/CADCOVID/Datasets_CoDAGANs/'
dataset = ImageFolder(data_root=root, n_datasets=3, mode='test', label_use='1|1|1', resize_to=(284, 284))
print(dataset.__getitem__(0))
'''