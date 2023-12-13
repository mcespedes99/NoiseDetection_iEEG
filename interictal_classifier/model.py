import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class NN(nn.Module):
    def __init__(self,NFILT=256,NOUT=4):
        super().__init__()
        self.conv0 = nn.Conv2d(1,NFILT,kernel_size=(200,3),padding=(0,1),bias=False) 
        # out: N,C,H,W but C=1. H=freq, W=time
        self.bn0 = nn.BatchNorm2d(NFILT)
        # GRU receives: N, L, H when batch_first=True
        self.gru = nn.GRU(input_size=NFILT,hidden_size=128,num_layers=1,batch_first=True,bidirectional=False)
        self.fc1 = nn.Linear(128,NOUT)



    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = x.squeeze().permute(0,2,1) # permuting from N, f, t to N, t, f
        x,_ = self.gru(x)
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.fc1(x)
        return x

# Create a convolutional neural network 
class IeegClassifier(nn.Module):
    """
    Model to classify ieeg data based on CNNs
    To calculate size: http://layer-calc.com/
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=32, 
                      kernel_size=5, # how big is the square that's going over the image?
                      stride=1, # default
                      padding='valid'),# output: 96 by 96 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2), # default stride value is same as kernel_size
            nn.Conv2d(in_channels=32, 
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.classifier = nn.Sequential(
            # nn.GRU(input_size=NFILT,hidden_size=128,num_layers=1,batch_first=True,bidirectional=False),
            nn.Flatten(),
            nn.Linear(in_features=64*22*22, 
                      out_features=1000),
            nn.Linear(in_features=1000, 
                      out_features=n_classes),  # 4 classes        
        )
    
    def forward(self, x: torch.Tensor):
        x = self.cnn_block(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
