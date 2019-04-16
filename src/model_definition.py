# encoding: utf-8

"""
Declare the model used in training and inference
Change the Model class if you want to change how the model works
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

class Model(nn.Module):
    """Simple AlexNet bidir-LSTM model"""
    def __init__(self, num_labels, device, final_height=18):
        super(Model, self).__init__()
        "modify this method to change the available network components"
        # final height is the image height at last conv layer output
        self.final_height = final_height
        self.num_labels = num_labels
        self.device = device
        # architecture params (f = features)
        self.num_conv1_f = 16
        self.num_conv2_f = 32
        self.num_conv3_f = 48
        self.num_conv4_f = 64
        self.num_lstm_f = 128
        self.num_lstm_layers = 1
        # these layers are imported when you load the pretrained model
        self.conv1 = nn.Conv2d(1, self.num_conv1_f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.num_conv1_f, self.num_conv2_f, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.num_conv2_f, self.num_conv3_f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.num_conv3_f, self.num_conv4_f, kernel_size=3, padding=1)
        self.lstm1 = nn.LSTM(input_size=self.num_conv4_f*self.final_height, 
            hidden_size=self.num_lstm_f,
            num_layers=self.num_lstm_layers,
            bidirectional=False,
            dropout=0.0)
        self.linear = nn.Linear(self.num_lstm_f, self.num_labels+1)
    def reset_lstm(self):
        """modify this method to declare which layers are reset or frozen for transfer learning"""
        # freeze layers with weights
        for param in self.parameters():
            param.requires_grad = False
        
        # redeclare layers to reset their weights 
        # self.lstm1 = nn.LSTM(input_size=self.num_conv4_f*self.final_height, 
        #     hidden_size=self.num_lstm_f,
        #     num_layers=self.num_lstm_layers,
        #     bidirectional=True,
        #     dropout=0.5)
        self.linear = nn.Linear(self.num_lstm_f, self.num_labels+1)

    def get_lstm_inits(self, batch_size):
        # generate inits so we can deal with variable batch sizes
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.num_lstm_f).to(self.device)
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.num_lstm_f).to(self.device)
        return h0, c0

    def forward(self, x, debug=False):
        """modify this method to change the model architecture"""
        x = torch.relu(self.conv1(x))
        x = F.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.relu(self.conv3(x))
        x = F.max_pool2d(torch.tanh(self.conv4(x)), 2)
        if debug: print("final height: ", x.shape[2])
        # (N, C, H, W)
        # stack channels as features in the H dimension
        x = x.view(x.shape[0], -1, x.shape[3])
        # (N, H*C, W)
        if debug: # set debug to True when calling the forward method to get some internal info 
            # convert to numpy
            im = x[0].detach().cpu().numpy()
            # normalise
            im = im - im.min()
            if im.max() != 0: # handle div by zero
                im = im/im.max()
            im = Image.fromarray(np.uint8(255*im))
            # save a feature file 
            im.save("features_viz.png")
            print(x.shape)
        x = x.permute(2,0,1)
        # (t, N, F)
        x, _ = self.lstm1(x, self.get_lstm_inits(x.shape[1]))
        # run an identical linear layer on all feature columns to produce logits
        # nn.Linear takes an arbitrary number of dimensions between the batch dimension and the last dimension
        x = self.linear(x.permute(1,0,2))
        x = x.permute(1,0,2)
        # (t, N, L) where L is num_labels+1
        x = F.log_softmax(x, 2)
        return x
