import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import librosa as lb
import numpy as np
import os
import random

TUT_LABELS = ["fan", "pump", "slider", "ToyCar", "ToyConveyor", "valve"]
TUT_LABELS_INT = [0, 7, 13, 20, 27, 34]

class TUTDataset(Dataset):

    def __init__(self, path_data, list_files, sample_rate, duration):
        # Initialization for path, fs, list of files etc. 
        self.sample_rate = sample_rate
        self.path_data = path_data
        self.list_files = list_files
        self.duration = duration

    def __getitem__(self, index):
        # select an audio from the list and return metadata and label
        file_name = self.list_files[index]
        if ".npy" in file_name:
            audio_data = np.load(file_name)
        else:
            audio_data , _ = lb.load(file_name, sr = self.sample_rate, res_type = "polyphase")
        if len(audio_data) > int(self.duration * self.sample_rate):
            audio_data = audio_data[:int(self.duration * self.sample_rate)]

        metadata = 0
        numerical_label = 0
        for i, name_class in enumerate(TUT_LABELS):
            if name_class in file_name:
                metadata = TUT_LABELS_INT[i]
                numerical_label = TUT_LABELS_INT[i]
                break

        metadata += int(file_name.split('_')[2])  # in this way I have "name_id", for example fan_01 where 01 is the id of the machine
        label = 0 if "normal" in file_name else 1
        return audio_data , metadata, label, numerical_label
    
    def __len__(self):
        # return length of the list
        return len(self.list_files)

class TUTDatamodule(LightningDataModule):

    def __init__(self, path_train, path_test, sample_rate, duration, percentage_val, batch_size):
        super().__init__()
        self.path_train = path_train
        self.path_test = path_test
        self.sample_rate = sample_rate
        self.duration = duration
        self.percentage_val = percentage_val
        self.batch_size = batch_size
        # split of the dataset
        self.train_list = self.scan_all_dir(self.path_train)
        self.val_list = random.sample(self.train_list, int(len(self.train_list) * percentage_val))
        self.train_list = set(self.train_list)
        self.val_list = set(self.val_list)
        self.train_list -= self.val_list
        self.train_list = list(self.train_list)
        self.val_list = list(self.val_list)

        self.test_list = self.scan_all_dir(self.path_test)


    def scan_all_dir(self, path):
        list_all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                list_all_files.append(str(root + "\\" + file))
        return list_all_files

    def setup(self, stage = None):
        # Nothing to do
        pass

    def prepare_data(self):
        # Nothing to do
        pass
    
    def train_dataloader(self):
        # return the dataloader containing training data
        train_split = TUTDataset(path_data = self.path_train, list_files = self.train_list,  sample_rate = self.sample_rate, duration = self.duration)
        return DataLoader(train_split, batch_size = self.batch_size, shuffle = True)
    
    def val_dataloader(self):
        # return the dataloader containing validation data
        val_split = TUTDataset(path_data = self.path_train, list_files =  self.val_list, sample_rate = self.sample_rate, duration = self.duration)
        return DataLoader(val_split, batch_size = self.batch_size, shuffle = False)
    
    def test_dataloader(self):
        # return the dataloader containing testing data
        test_split = TUTDataset(path_data = self.path_test, list_files = self.test_list, sample_rate = self.sample_rate, duration = self.duration)
        return DataLoader(test_split, batch_size = self.batch_size, shuffle = True)
    
## TEST FUNCTION ##
if __name__ == "__main__":
    path_train = "TUT Anomaly detection/train" # path for training audio
    path_test = "TUT Anomaly detection/test" # path for test audio
    percentage_val = 0.2
    sample_rate = 16000
    batch_size = 64
    duration = 10
    # create lightning datamodule
    datamodule = TUTDatamodule(path_train = path_train, path_test = path_test, sample_rate = sample_rate, duration = duration,
                                percentage_val = percentage_val, batch_size = batch_size)
    dataloader_train = datamodule.train_dataloader()
    print(len(dataloader_train))
    dataloader_val = datamodule.val_dataloader()
    print(len(dataloader_val))
    dataloader_test = datamodule.test_dataloader()
    print(len(dataloader_test))
    print(next(iter(dataloader_test)))