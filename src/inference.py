# encoding: utf-8

"""
Script to load your trained model and create a submission as a csv file
"""

import os
import csv
import numpy as np
import json
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model_definition import Model
from read_data import HW_Dataset
import utils

# Global Vars
CLASS_NAMES = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
]
NUM_CLASSES = len(CLASS_NAMES)

DATA_PHASE = 'phase_2'
LABELS_DIR = '../%s/labels'%DATA_PHASE
IMGS_DIR = '../%s/images'%DATA_PHASE
# when you get the phase 2 data comment the line below and
#  uncomment the next line
TEST_LIST = '../phase_2/test_list.json'
# TEST_LIST = '../phase_2/test_list.json'
PRIOR_MODEL_PATH = '../models/hw_model_ref.pth'
LOG_INTERVAL = 100
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
SUBMISSION_PATH = "../submission.csv"

USE_GPU = True
GPU_IDX = 0

# Load model
device = torch.device('cuda:%s'%GPU_IDX if (USE_GPU and torch.cuda.is_available()) else "cpu")
model = Model(NUM_CLASSES, device).to(device)
model.load_state_dict(torch.load(PRIOR_MODEL_PATH, map_location=device))

# Load and predict data
normalize = transforms.Normalize(mean=[0.000],
                                std=[1.000])
test_dataset = HW_Dataset(labels_dir=LABELS_DIR,
                            imgs_dir=IMGS_DIR,
                            data_list=TEST_LIST,
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
        if batch_idx % LOG_INTERVAL == 0:
            print("performed inference on batch: ", batch_idx)
        x = x.to(device)
        y = model(x)
        y_target = utils.encode_words(y_target, CLASS_NAMES)
        preds.extend(utils.decode_output(y.cpu(), CLASS_NAMES))
        targets.extend(utils.decode_label_words(y_target, CLASS_NAMES))
    # save submission
    utils.save_predictions(SUBMISSION_PATH, TEST_LIST, preds)
