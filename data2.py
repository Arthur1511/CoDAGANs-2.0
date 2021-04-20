import os
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from skimage import io, transform
from sklearn.utils.class_weight import (compute_class_weight,
                                        compute_sample_weight)


def default_loader(path):

    return io.imread(path)


def get_img_list(dir, dataset_number, csv_file, mode, label_use=True,):
    img_list = []
    label_list = []
    diseases = ['cardiomegaly', 'atelectasis', 'pneumonia']
    num_labels = len(diseases)

    img_dir = os.path.join(dir, 'dataset' + str(dataset_number) + 'images/')
    # csv_labels = str(dir + 'dataset' + dataset_number + '/' + csv_file)
    labels = pd.read_csv(dir + 'dataset' + str(dataset_number) + '/' +
                         csv_file, usecols=['Image_ID', 'Labels'])

    tam_dataset = labels.shape[0]

    if mode == 'train':
        tam_list = int(0.8*tam_dataset)
        labels = labels.iloc[0:tam_list]
        # print("Train len: ", len(labels))
    else:
        tam_list = int(0.2*tam_dataset)
        labels = labels.iloc[-tam_list:]
        # print("Test len: ", len(labels))

    class_count = [0, 0, 0, 0]

    for i in range(tam_list):

        if label_use == False:
            img = labels.iloc[i]['Image_ID']
            img_list.append(img)
            label_list.append(-1)

        else:
            lab = labels.iloc[i]['Labels']
            img = labels.iloc[i]['Image_ID']
            try:
                lab = lab.lower().split('|')
            except:
                lab = ['no finding']

            labels_numbers = sum(map(lambda x: x in diseases, lab))
            # if not all(x in lab for x in ['cardiomegaly', 'atelectasis']):
            if labels_numbers < 2:
                #             label_list.append(lab)
                # if set(lab) == '':
                #     label_list.append(-1)
                no_finding = True

                for i, label in enumerate(diseases):

                    if label in lab:
                        label_list.append(i)
                        class_count[i] += 1
                        no_finding = False

                if no_finding:
                    i = num_labels
                    label_list.append(num_labels)
                    class_count[num_labels] += 1

                img_list.append(img)

                # if 'cardiomegaly' in lab:
                #     label_list.append(0)
                #     class_count[0] += 1
                # elif 'atelectasis' in lab:
                #     label_list.append(1)
                #     class_count[1] += 1
                # elif 'pneumonia' in lab:
                #     label_list.append(2)
                #     class_count[2] += 1
                # else:
                #     label_list.append(3)
                #     class_count[3] += 1

                # img_list.append(img)

    return img_list, label_list, class_count


# number = '0'
# list_im, list_label, class_count = get_img_list(
#     '../CADCOVID/Datasets_CoDAGANs/', number, 'dataset' + number + '.csv', 'train')
# print(np.unique(list_label, return_counts=True))
# print("Class count: ", class_count)
# print(list_label[:10])


class ImageFolder(data.Dataset):

    def __init__(self, root, n_datasets, label_use, mode='train', resize_to=(256, 256), return_path=False):

        dataset_numbers = list(range(n_datasets))
        datasets_folders = ['dataset' + str(num) for num in dataset_numbers]
        files_csv = [d + '.csv' for d in datasets_folders]

        imgs = list()
        labels = list()
        classes_count = list()

        for i in range(n_datasets):
            img, label, class_count = get_img_list(
                root, i, files_csv[i], mode, label_use=label_use[i])
            imgs.append(img)
            labels.append(label)
            classes_count.append(class_count)

        self.root = root
        self.datasets_folders = datasets_folders
        self.imgs = imgs
        self.labels = labels
        self.mode = mode
        self.return_path = return_path
        self.label_use = label_use
        self.resize_to = resize_to
        self.n_classes = len(np.unique(labels))
        self.classes_count = np.array(classes_count)
        # self.class_count = np.array(class_count, dtype=np.float)
        # self.samples_weights = compute_sample_weight('balanced', y=labels)
        # self.weight_class = np.divide(
        #     float(len(labels)), (self.n_classes * self.class_count), out=np.zeros_like(self.class_count), where=self.class_count != 0)
        # self.samples_weights = self.weight_class[self.labels]
        # self.class_count = class_count
        # self.loader = loader
        # self.sample = sample
        # self.trim_bool = trim_bool
        # self.random_transform = random_transform
        # self.channels = channels
        # self.normalization = normalization
        # self.weight_class = len(
        #     labels) / (self.n_classes * self.class_count)

    def load_samples(self, n_samples, d_index):

        sample_list = [self.load_preprocess(
            self.imgs[d_index][i], d_index) for i in range(n_samples)]

        img_list = [s[0] for s in sample_list]
        label_list = [self.labels[d_index][i] for i in range(n_samples)]
        lbl_list = [s for s in label_list]

        return img_list, lbl_list

    def load_preprocess(self, img_path, d_index):

        # Loading.
        file_name = str(self.root +
                        self.datasets_folders[d_index] + '/images/' + img_path)
        img = io.imread(file_name)

        # Resizing.
        img = transform.resize(img, self.resize_to,
                               order=1, preserve_range=True)

        # Normalization and transformation to tensor.
        img = img.astype(np.float32)
        # img = (img - img.mean()) / (img.std() + 1e-10)
        img = (img - img.min()) / (img.max() - img.min() + 1e-10) - 0.5
        img = np.expand_dims(img, axis=0)

        if len(img.shape) != 3:
            img = img[:, :, :, 0]

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

        # print(ind_a)

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


# input_folder = '../CADCOVID/Datasets_CoDAGANs/'
# for i in range(4):
# dataset = ImageFolder(input_folder, sample=1, n_datasets=2, mode='train', label_use=[True, True, True], trim_bool=1,
#                       return_path=True, random_transform=2, channels=1)
# print(dataset.classes_count)

# for i in range(dataset.__len__()):
# print(dataset.__getitem__(0))
