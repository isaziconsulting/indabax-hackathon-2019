# encoding: utf-8

"""
Script to train handwriting model

Change global vars if you want to change how data is loaded or to change major model training params
"""

import os
import json
import time
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image

from model_definition import Model
from read_data import HW_Dataset
import utils

# Global Vars
CLASS_NAMES = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
]
NUM_CLASSES = len(CLASS_NAMES)

LABELS_DIR = '../phase_1/labels'
IMGS_DIR = '../phase_1/images'
TRAIN_LIST = '../phase_1/train_list.json'
VAL_LIST = '../phase_1/val_list.json'
SAVE_MODEL_PATH = '../models/hw_model.pth'
PRIOR_MODEL_PATH = '../models/hw_model_ref.pth'
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8

LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
BATCH_LOG_INTERVAL = 100
USE_GPU = True
GPU_IDX = 0
USE_PRIOR = False # transfer learning
SAVE_MODEL = True
STOPPING_LOSS = 0.25

# Initialize and Load the Model
device = torch.device('cuda:%s'%GPU_IDX if (USE_GPU and torch.cuda.is_available()) else "cpu")
model = Model(NUM_CLASSES, device).to(device)
if USE_PRIOR:
    model.load_state_dict(torch.load(PRIOR_MODEL_PATH))
    model.reset_lstm()
    model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, threshold=1e-3)

# Define Train and Test Functions
def train(start_time, epoch):
    model.train()

    normalize = transforms.Normalize(mean=[0.000],
                                    std=[1.000])
    train_dataset = HW_Dataset(labels_dir=LABELS_DIR,
                                    imgs_dir=IMGS_DIR,
                                    data_list=TRAIN_LIST,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE,
                                shuffle=False, num_workers=8, pin_memory=True)
    model.train()
    epoch_loss = 0
    start = time.time()
    for batch_idx, (x, y_target) in enumerate(train_loader):
        # handle gpu
        x = x.to(device)
        # foward propagate
        y = model(x)
        # encode target for ctc
        y_target = utils.encode_words(y_target, CLASS_NAMES)
        # pytorch ctc loss
        input_lengths = torch.tensor([y.shape[0]]*y.shape[1], dtype=torch.long).to(device)
        target_lengths = torch.tensor([len(l) for l in y_target], dtype=torch.long).to(device)
        y_target = torch.cat(y_target)
        loss = F.ctc_loss(y, y_target, input_lengths, target_lengths, reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        if batch_idx > 0 and batch_idx % BATCH_LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime per batch: {:.2f} s\tTotal Time: {:.2f} hrs'.format(
                epoch, batch_idx * TRAIN_BATCH_SIZE, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), (time.time() - start) / BATCH_LOG_INTERVAL, (time.time()-start_time)/3600))
            start = time.time()

    print('Avg. Epoch Loss: {:.6f}\n'.format(epoch_loss/len(train_loader)))
    scheduler.step(epoch_loss)

def test(print_sample_preds=True):
    normalize = transforms.Normalize(mean=[0.000],
                                    std=[1.000])
    test_dataset = HW_Dataset(labels_dir=LABELS_DIR,
                                imgs_dir=IMGS_DIR,
                                data_list=VAL_LIST,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=TEST_BATCH_SIZE,
                                shuffle=False, num_workers=0, pin_memory=True)
    # switch to evaluate mode (deactivate dropout)
    model.eval()
    test_loss = 0
    num_incorrect = 0
    total_str_dist = 0
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx, (x, y_target) in enumerate(test_loader):
            x = x.to(device)
            y = model(x)
            y_target = utils.encode_words(y_target, CLASS_NAMES)
            # pytorch ctc loss
            input_lengths = torch.tensor([y.shape[0]]*y.shape[1], dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(l) for l in y_target], dtype=torch.long).to(device)
            loss = F.ctc_loss(y, torch.cat(y_target), input_lengths, target_lengths, reduction='mean')
            test_loss += loss.item()
            preds.extend(utils.decode_output(y.cpu(), CLASS_NAMES))
            targets.extend(utils.decode_label_words(y_target, CLASS_NAMES))
            if print_sample_preds and batch_idx == 0:
                print("Test Prediction")
                print(preds)
                print("Test Ground Truth")
                print(targets, "\n")
        num_incorrect += sum([p!=gt for gt,p in zip(targets, preds)]) # WER
        total_str_dist += sum([utils.norm_levenshtein_dist(x,y) for x,y in zip(targets, preds)]) # CER
        # calculate accuracy stats
        WER = 100*(num_incorrect/(len(test_dataset)))
        CER = 100*(total_str_dist/(len(test_dataset)))
        test_loss = test_loss*TEST_BATCH_SIZE/len(test_dataset)
        print('Avg. Test Loss: {:.4f}\tCER: {:.1f}%\tWER: {}/{} ({:.1f}%) \n'.format(
            test_loss, CER, num_incorrect, len(test_dataset), WER))
        # termination condition
        finished_training = test_loss <= STOPPING_LOSS
        return finished_training

# Run Train and Test
if __name__ == '__main__':
    start_time = time.time()
    for epoch in range(MAX_EPOCHS):
        train(start_time, epoch)
        # has termination condition been met
        if test(print_sample_preds=True):
            break
    if SAVE_MODEL:
        utils.save_model(model, SAVE_MODEL_PATH)
