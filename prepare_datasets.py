import matplotlib.pyplot as plt
import numpy as np
import h5py
import librosa
import os, sys
import time
import pandas as pd 
from collections import namedtuple
from sklearn.model_selection import StratifiedShuffleSplit

PATH_DATASETS = '/home/bhubon/fine_tuning_genre/genre_221216_docker_project/dataset/'
#FOLDER_CSV = 'F:\\CATMood_Dataset\\CATMood Dataset'
FOLDER_CSV = '/home/bhubon/fine_tuning_genre/genre_221216_docker_project/data_csv/'

allowed_exts = set(['mp3', 'wav', 'au'])
column_names = ['id', 'filepath', 'label'] # todo: label_for_stratify?

def write_to_csv(rows, column_names, csv_fname):
    '''rows: list of rows (= which are lists.)
    column_names: names for columns
    csv_fname: string, csv file name'''
    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(os.path.join(FOLDER_CSV, csv_fname))
    
def get_rows_from_folders(folder_dataset, folders, dataroot=None):
    '''gtzan, ballroom extended. each class in different folders'''
    rows = []
    if dataroot is None:
        dataroot = PATH_DATASETS
    for label_idx, folder in enumerate(folders): # assumes different labels per folders.
        files = os.listdir(os.path.join(dataroot, folder_dataset, folder))
        files = [f for f in files if f.split('.')[-1].lower() in allowed_exts]
        for fname in files:
            file_path = os.path.join(folder_dataset, folder, fname)
            file_id =fname.split('.')[0]
            file_label = label_idx
            rows.append([file_id, file_path, file_label])
    print('Done - length:{}'.format(len(rows)))
    print(rows[0])
    print(rows[-1])
    return rows

### For gtzan genre


folder_dataset_gtg = '/home/bhubon/fine_tuning_genre/genre_221216_docker_project/dataset/'
labels_gtg = ['Court_music_C', 'Creative_gugak_R', 'Folk_music_F', 'Fusion_gugak_S', 'Pungryu_music_E']
n_label_gtg = len(labels_gtg)
folders_gtg = [s + '/' for s in labels_gtg]

rows_gtg = get_rows_from_folders(folder_dataset_gtg, folders_gtg)
write_to_csv(rows_gtg, column_names, 'genre_221216.csv')
