import torch
import torch.nn as nn

class ConvReluLayer(nn.Module):
    """Convolutional layer with ReLU activation function (and optional batch normalization)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, batch_norm=False):
        super(ConvReluLayer, self).__init__()
        self.layer = nn.Sequential()
        
        self.layer.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        if batch_norm: self.layer.add_module('batchnorm', nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layer.add_module('relu', nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.layer(x)

# Possible error: First convolution needs to be coordconv
class STR_Model(nn.Module):
    '''Main Stroke Trajectory Recovery model. Takes in images on size 60 x W and outputs a sequence of 4D vectors of size W/4 x 4'''
    def __init__(self):
        super(STR_Model, self).__init__()
    
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.max_pool_2x1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)
        self.cnn_branch = nn.Sequential( # Input: batch x 1 x 60 x W
            ConvReluLayer(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),    # b x 64 x 60 x W
            self.max_pool_2x2,                                                          # b x 64 x 30 x W/2
            ConvReluLayer(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # b x 128 x 30 x W/2
            self.max_pool_2x2,                                                          # b x 128 x 15 x W/4
            ConvReluLayer(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), batch_norm=True), # b x 256 x 15 x W/4
            ConvReluLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # b x 256 x 15 x W/4
            self.max_pool_2x1,                                                          # b x 256 x 7 x W/4 + 1
            ConvReluLayer(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), batch_norm=True), # b x 512 x 7 x W/4
            ConvReluLayer(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # b x 512 x 7 x W/4
            self.max_pool_2x1,                                                          # b x 512 x 3 x W/4 + 2
            ConvReluLayer(512, 512, kernel_size=(2, 2), stride=(1, 1), batch_norm=True),# b x 512 x 2 x W/4 + 1
        )
        
        # Input: W/4 x batch x 1024
        self.rnn_branch = nn.LSTM(input_size=1024, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.5)
        
        self.embedding = nn.Linear(in_features=2*128, out_features=4) # Output: batch x T x (x, y, SoS, EoS)
        
    def postprocess_cnn(self, x):
        '''Input: batch x 512 x 2 x W/4
           Output: W/4 x batch x 1024'''
        b, c, h, w = x.size() # b x 512 x 2 x W/4
        x = x.view(b, c*h, w) # b x 1024 x W/4
        
        # Make width the "time" seq2seq dimension
        x = x.permute(2, 0, 1) # W/4 x b x 1024
        return x
        
    def forward(self, x):
        '''Input: batch x 1 x 60 x W
           Output: W/4 x batch x 4'''
        x = self.cnn_branch(x)
        x = self.postprocess_cnn(x)
        x, _ = self.rnn_branch(x) # Output: W/4 x b x 2*128
        T, b, h = x.size()        # W/4 x b x 2*128
        x = x.view(T * b, h)
        x = self.embedding(x)     # Output: W/4*b x 4
        x = x.view(T, b, 4)    # Output: W/4 x b x 4
        # x is relative cordinates. Make it absolute by taking the cumulative sum
        x[:, :, :2] = torch.cumsum(x[:, :, :2], dim=0) # Out
        return x
        
class STR_Model_Longer_512(nn.Module):
    '''Stroke Trajectory Recovery model that outputs a sequence of 4D vectors of size W x 4'''
    def __init__(self):
        super(STR_Model_Longer_512, self).__init__()
    
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.max_pool_2x1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)
        self.cnn_branch = nn.Sequential( # Input: batch x 1 x 60 x W
            ConvReluLayer(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),    # b x 64 x 60 x W
            self.max_pool_2x2,                                                          # b x 64 x 30 x W/2
            ConvReluLayer(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # b x 128 x 30 x W/2
            self.max_pool_2x1,                                                          # b x 128 x 15 x W/2 + 1
            ConvReluLayer(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), batch_norm=True), # b x 256 x 15 x W/2 + 1
            ConvReluLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # b x 256 x 15 x W/2 + 1
            self.max_pool_2x1,                                                          # b x 256 x 7 x W/2 + 2
            ConvReluLayer(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0), batch_norm=True), # b x 512 x 7 x W/2
            ConvReluLayer(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # b x 512 x 7 x W/2
            self.max_pool_2x1,                                                          # b x 512 x 3 x W/2 + 1
            ConvReluLayer(512, 512, kernel_size=(2, 2), stride=(1, 1), batch_norm=True),# b x 512 x 2 x W/2
        )
        
        # Input: W x batch x 512
        self.rnn_branch = nn.LSTM(input_size=512, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.5)
        
        self.embedding = nn.Linear(in_features=2*128, out_features=4) # Output: batch x T x (x, y, SoS, EoS)
        
    def postprocess_cnn(self, x):
        '''Input: batch x 512 x 2 x W/2
           Output: W x batch x 512'''
        b, c, h, w = x.size() # b x 512 x 2 x W/2
        x = x.view(b, c, h*w) # b x 512 x W
        
        # Make width the "time" seq2seq dimension
        x = x.permute(2, 0, 1) # W x b x 512
        return x
    
    def forward(self, x):
        '''Input: batch x 1 x 60 x W
           Output: W x batch x 4'''
        x = self.cnn_branch(x)
        x = self.postprocess_cnn(x)
        x, _ = self.rnn_branch(x)
        T, b, h = x.size()
        x = x.view(T * b, h)
        x = self.embedding(x)
        # return x.view(T, b, 4)
        x = x.view(T, b, 4)    # Output: W x b x 4
        # x is relative cordinates. Make it absolute by taking the cumulative sum
        x[:, :, :2] = torch.cumsum(x[:, :, :2], dim=0) # Out
        return x
    
class STR_Model_Longer_1024(nn.Module):
    '''Stroke Trajectory Recovery model that outputs a sequence of 4D vectors of size W x 4'''
    def __init__(self):
        super(STR_Model_Longer_1024, self).__init__()
    
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.max_pool_2x1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)
        self.cnn_branch = nn.Sequential( # Input: batch x 1 x 60 x W
            ConvReluLayer(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),    # b x 64 x 60 x W
            self.max_pool_2x2,                                                          # b x 64 x 30 x W/2
            ConvReluLayer(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # b x 128 x 30 x W/2
            self.max_pool_2x1,                                                          # b x 128 x 15 x W/2 + 1
            ConvReluLayer(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), batch_norm=True), # b x 256 x 15 x W/2 + 1
            ConvReluLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # b x 256 x 15 x W/2 + 1
            self.max_pool_2x1,                                                          # b x 256 x 7 x W/2 + 2
            ConvReluLayer(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0), batch_norm=True), # b x 512 x 7 x W/2
            ConvReluLayer(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # b x 512 x 7 x W/2
            self.max_pool_2x1,                                                          # b x 512 x 3 x W/2 + 1
            ConvReluLayer(512, 1024, kernel_size=(2, 2), stride=(1, 1), batch_norm=True)# b x 1024 x 2 x W/2
        )
        
        # Input: W x batch x 1024
        self.rnn_branch = nn.LSTM(input_size=1024, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.5)
        
        self.embedding = nn.Linear(in_features=2*128, out_features=4)
        
    def postprocess_cnn(self, x):
        '''Input: batch x 1024 x 2 x W/2
           Output: W x batch x 1024'''
        b, c, h, w = x.size()
        x = x.view(b, c, w*h)
        
        # Make width the "time" seq2seq dimension
        x = x.permute(2, 0, 1)
        return x
    
    def forward(self, x):
        '''Input: batch x 1 x 60 x W
           Output: W x batch x 4'''
        x = self.cnn_branch(x)
        x = self.postprocess_cnn(x)
        x, _ = self.rnn_branch(x)
        T, b, h = x.size()
        x = x.view(T * b, h)
        x = self.embedding(x)
        # return x.view(T, b, 4)
        x = x.view(T, b, 4)    # Output: W x b x 4
        # x is relative cordinates. Make it absolute by taking the cumulative sum
        x[:, :, :2] = torch.cumsum(x[:, :, :2], dim=0)
        return x