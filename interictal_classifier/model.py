import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class NN(nn.Module):
    def __init__(self,NFILT=256,NOUT=4):
        super().__init__()
        self.conv0 = nn.Conv2d(1,NFILT,kernel_size=(100,3),padding=(0,1),bias=False) 
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

class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int, input_size: int):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 16, kernel_size=3, padding='valid')
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding='valid')
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32 * 23 * 23, n_classes),
            # nn.ReLU()
        )
        # self.fc2 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(128, n_classes)
        # )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # x = self.relu(self.fc1(x))
        x = self.fc1(x)
        return x

class SimpleCNN2(nn.Module):
    def __init__(self, n_classes: int, input_size: int):
        super(SimpleCNN2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, 8, kernel_size=3, padding='valid'),
            nn.BatchNorm2d(8),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding='valid'),
            nn.BatchNorm2d(16),
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16 * 23 * 23, n_classes),
            # nn.ReLU()
        )
        # self.fc2 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(128, n_classes)
        # )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # x = self.relu(self.fc1(x))
        x = self.fc1(x)
        return x

# Create a convolutional neural network 
class MultiLayerClassifier(nn.Module):
    """
    Model to classify ieeg data based on CNNs
    To calculate size: http://layer-calc.com/
    """
    def __init__(self, n_classes: int, input_size: int):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, 
                      out_channels=16, 
                      kernel_size=5, # how big is the square that's going over the image?
                      stride=1, # default
                      padding='same'),# output: 100 by 100
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=16, 
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.cnn_rect = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, 
                      out_channels=32,
                      kernel_size=(50,3),
                      stride=1,
                      padding=(0,1)),
            nn.BatchNorm2d(32),
        )
        self.gru =  nn.LSTM(input_size=32,hidden_size=32,num_layers=1,batch_first=True,bidirectional=False, dropout=0.5)
        self.classifier = nn.Linear(in_features=32, out_features=n_classes)
        #nn.Sequential(
            # nn.Linear(in_features=64, 
            #           out_features=1000),
            # GRU receives: N, L, H when batch_first=True
            # nn.GRU(input_size=128,hidden_size=64,num_layers=1,batch_first=True,bidirectional=False),
        #     nn.Linear(in_features=64, 
        #               out_features=n_classes),  # 4 classes        
        # )
    
    def forward(self, x: torch.Tensor):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.cnn_rect(x)
        # print(x.shape)
        x = x.squeeze(dim=2).permute(0,2,1) # permuting from N, f, t to N, t, f
        x,_ = self.gru(x)
        x = F.dropout(x,p=0.5,training=self.training)
        # print(x.shape)
        x = self.classifier(x)[:,-1,:]
        # print(x.shape)
        return x

# Create a convolutional neural network 
class MultiLayerCNN(nn.Module):
    """
    Model to classify ieeg data based on CNNs
    To calculate size: http://layer-calc.com/
    """
    def __init__(self, n_classes: int, input_size: int):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, 
                      out_channels=16, 
                      kernel_size=5, # how big is the square that's going over the image?
                      stride=1, # default
                      padding='same'),# output: 100 by 100
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=16, 
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.cnn3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, 
                      out_channels=32,
                      kernel_size=3,
                      padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, 
                      out_channels=32,
                      kernel_size=3,
                      padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(0.5),
            # nn.Linear(in_features=32*11*11, out_features=1000),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=32*11*11, out_features=n_classes),
        )
        #nn.Sequential(
            # nn.Linear(in_features=64, 
            #           out_features=1000),
            # GRU receives: N, L, H when batch_first=True
            # nn.GRU(input_size=128,hidden_size=64,num_layers=1,batch_first=True,bidirectional=False),
        #     nn.Linear(in_features=64, 
        #               out_features=n_classes),  # 4 classes        
        # )
    
    def forward(self, x: torch.Tensor):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.cnn3(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

# Create a convolutional neural network 
class MultiLayerCNNSmall(nn.Module):
    """
    Model to classify ieeg data based on CNNs
    To calculate size: http://layer-calc.com/
    """
    def __init__(self, n_classes: int, input_size: int):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, 
                      out_channels=16, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding='same'),# output: 100 by 100
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, 
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.cnn3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, 
                      out_channels=64,
                      kernel_size=5,
                      padding='valid'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(0.5),
            # nn.Linear(in_features=64*10*10, out_features=1000),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=64*10*10, out_features=n_classes),
        )
        #nn.Sequential(
            # nn.Linear(in_features=64, 
            #           out_features=1000),
            # GRU receives: N, L, H when batch_first=True
            # nn.GRU(input_size=128,hidden_size=64,num_layers=1,batch_first=True,bidirectional=False),
        #     nn.Linear(in_features=64, 
        #               out_features=n_classes),  # 4 classes        
        # )
    
    def forward(self, x: torch.Tensor):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x1)
        # Apply pool to x1
        pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        x = torch.cat([pooling(x1), x2], dim=1)
        x = self.cnn3(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

# Create a convolutional neural network 
class MultiLayerCNNSmall2(nn.Module):
    """
    Model to classify ieeg data based on CNNs
    To calculate size: http://layer-calc.com/
    """
    def __init__(self, n_classes: int, input_size: int):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, 
                      out_channels=16, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding='valid'),# output: 100 by 100
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, 
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.cnn3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, 
                      out_channels=64,
                      kernel_size=5,
                      padding='valid'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(0.5),
            # nn.Linear(in_features=64*9*9, out_features=500),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=64*9*9, out_features=n_classes),
        )
        #nn.Sequential(
            # nn.Linear(in_features=64, 
            #           out_features=1000),
            # GRU receives: N, L, H when batch_first=True
            # nn.GRU(input_size=128,hidden_size=64,num_layers=1,batch_first=True,bidirectional=False),
        #     nn.Linear(in_features=64, 
        #               out_features=n_classes),  # 4 classes        
        # )
    
    def forward(self, x: torch.Tensor):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x1)
        # Apply pool to x1
        x = self.cnn3(x2)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

# Create a convolutional neural network 
class MultiLayerCNNTiny(nn.Module):
    """
    Model to classify ieeg data based on CNNs
    To calculate size: http://layer-calc.com/
    """
    def __init__(self, n_classes: int, input_size: int):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, 
                      out_channels=8, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding='valid'),# output: 100 by 100
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, 
                      out_channels=8,
                      kernel_size=3,
                      stride=1,
                      padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.cnn3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, 
                      out_channels=32,
                      kernel_size=3,
                      padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(0.5),
            # nn.Linear(in_features=32*11*11, out_features=100),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=32*11*11, out_features=n_classes),
        )
        #nn.Sequential(
            # nn.Linear(in_features=64, 
            #           out_features=1000),
            # GRU receives: N, L, H when batch_first=True
            # nn.GRU(input_size=128,hidden_size=64,num_layers=1,batch_first=True,bidirectional=False),
        #     nn.Linear(in_features=64, 
        #               out_features=n_classes),  # 4 classes        
        # )
    
    def forward(self, x: torch.Tensor):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x1)
        pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        x = torch.cat([pooling(x1), x2], dim=1)
        # Apply pool to x1
        x = self.cnn3(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

# Create a convolutional neural network 
class IeegClassifier(nn.Module):
    """
    Model to classify ieeg data based on CNNs
    To calculate size: http://layer-calc.com/
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=8, 
                      kernel_size=5, # how big is the square that's going over the image?
                      stride=1, # default
                      padding='same'),# output: 100 by 100
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=8, 
                      out_channels=8,
                      kernel_size=5,
                      stride=1,
                      padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.cnn_rect = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, 
                      out_channels=32,
                      kernel_size=(50,3),
                      stride=1,
                      padding=(0,1)),
            nn.BatchNorm2d(32),
        )
        self.gru =  nn.LSTM(input_size=32,hidden_size=32,num_layers=1,batch_first=True,bidirectional=False, dropout=0.5)
        self.classifier = nn.Linear(in_features=32, out_features=n_classes)
        #nn.Sequential(
            # nn.Linear(in_features=64, 
            #           out_features=1000),
            # GRU receives: N, L, H when batch_first=True
            # nn.GRU(input_size=128,hidden_size=64,num_layers=1,batch_first=True,bidirectional=False),
        #     nn.Linear(in_features=64, 
        #               out_features=n_classes),  # 4 classes        
        # )
    
    def forward(self, x: torch.Tensor):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.cnn_rect(x)
        # print(x.shape)
        x = x.squeeze(dim=2).permute(0,2,1) # permuting from N, f, t to N, t, f
        x,_ = self.gru(x)
        x = F.dropout(x,p=0.5,training=self.training)
        # print(x.shape)
        x = self.classifier(x)[:,-1,:]
        # print(x.shape)
        return x
    
# Create a convolutional neural network 
class IeegClassifier2(nn.Module):
    """
    Model to classify ieeg data based on CNNs
    To calculate size: http://layer-calc.com/
    """
    def __init__(self, n_classes: int):
        super().__init__()
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 100x100x1
        self.e11 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32)  # output: 100x100x32
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 50x50x32

        # input: 284x284x64
        self.e21 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64)  # output: 50x50x64
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 25x25x64

        # input: 140x140x128
        self.e31 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128)  # output: 25x25x128
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # output: 50x50x64
        self.d11 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64)  # output: 50x50x64
        )

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # output: 100x100x64
        self.d21 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32)  # output: 100x100x32
        )

        self.flatten = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, 
                      out_channels=64,
                      kernel_size=(50,1),
                      stride=1,
                      padding=(0,1)),
            nn.BatchNorm2d(64),
        )
        self.gru =  nn.GRU(input_size=64,hidden_size=32,num_layers=1,batch_first=True,bidirectional=False)
        self.classifier = nn.Linear(in_features=32, out_features=n_classes)
    
    def forward(self, x: torch.Tensor):
        # Encoder
        xe11 = F.relu(self.e11(x))
        # print(xe11.shape)
        xp1 = self.pool1(xe11)
        # print(xp1.shape)

        xe21 = F.relu(self.e21(xp1))
        # print(xe21.shape)
        xp2 = self.pool2(xe21)
        # print(xp2.shape)

        xe31 = F.relu(self.e31(xp2))
        # print(xe31.shape)

        # Decoder
        xu1 = self.upconv1(xe31)
        # print(xu1.shape)
        xu11 = torch.cat([xu1, xe21], dim=1)
        xd11 = F.relu(self.d11(xu11))

        xu2 = self.upconv2(xd11)
        xu21 = torch.cat([xu2, xe11], dim=1)
        xd21 = F.relu(self.d21(xu21))

        # Flatten
        x = self.flatten(xd21)

        # print(x.shape)
        x = x.squeeze(dim=2).permute(0,2,1) # permuting from N, f, t to N, t, f
        x,_ = self.gru(x)
        x = F.dropout(x,p=0.5,training=self.training)
        # print(x.shape)
        x = self.classifier(x)[:,-1,:]
        # print(x.shape)
        return x

# Create a convolutional neural network 
class IeegClassifier3(nn.Module):
    """
    Model to classify ieeg data based on CNNs
    To calculate size: http://layer-calc.com/
    """
    def __init__(self, n_classes: int):
        super().__init__()
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 100x100x1
        self.e11 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding='same'),
             nn.BatchNorm2d(32)  # output: 100x100x32
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 50x50x32

        # input: 284x284x64
        self.e21 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64) # output: 50x50x64
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # output: 100x100x64
        self.d11 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding='same'), # output: 100x100x32
            nn.BatchNorm2d(32)
        )

        self.flatten = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, 
                      out_channels=64,
                      kernel_size=(50,1),
                      stride=1,
                      padding=(0,1)),
            nn.BatchNorm2d(64),
        )
        self.gru =  nn.GRU(input_size=64,hidden_size=32,num_layers=1,batch_first=True,bidirectional=False)
        self.classifier = nn.Linear(in_features=32, out_features=n_classes)
    
    def forward(self, x: torch.Tensor):
        # Encoder
        xe11 = F.relu(self.e11(x))
        # print(xe11.shape)
        xp1 = self.pool1(xe11)
        # print(xp1.shape)

        xe21 = F.relu(self.e21(xp1))
        # print(xe21.shape)
        # Decoder
        xu1 = self.upconv1(xe21)
        # print(xu1.shape)
        xu11 = torch.cat([xu1, xe11], dim=1)
        xd11 = F.relu(self.d11(xu11))
        # print(xd11.shape)
        # Flatten
        x = self.flatten(xd11)
        # print(x.shape)
        # print(x.shape)
        x = x.squeeze(dim=2).permute(0,2,1) # permuting from N, f, t to N, t, f
        x,_ = self.gru(x)
        x = F.dropout(x,p=0.5,training=self.training)
        # print(x.shape)
        x = self.classifier(x)[:,-1,:]
        # print(x.shape)
        return x


# Create a modified resnet
def custom_resnet34(n_classes: int, input_size: int, all_trainable:bool=False):
    from torchvision.models import resnet34, ResNet34_Weights
    # Initialize model
    weights = ResNet34_Weights.DEFAULT # .DEFAULT = best available weights 
    resnet_model = resnet34(weights=weights)#(pretrained=True)
    
    # Turn of gradient for parameters
    if not all_trainable:
        for param in resnet_model.parameters():
            param.requires_grad = False
    
    # Modify layers
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Change first layer to accept 1 channel with the right shape
    resnet_model.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=input_size, 
                            out_channels=64, 
                            kernel_size=2, # how big is the square that's going over the image?
                            stride=1, # default
                            padding='valid'
                        ),# output: 96 by 96 
    )
    num_ftrs = resnet_model.fc.in_features
    out_ftrs = resnet_model.fc.out_features
    resnet_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=num_ftrs, 
                      out_features=n_classes),  # 4 classes        
        )
    #nn.Linear(num_ftrs, n_classes)

    return resnet_model

# Create a convolutional neural network 
# Old implementation
# class IeegClassifier(nn.Module):
#     """
#     Model to classify ieeg data based on CNNs
#     To calculate size: http://layer-calc.com/
#     """
#     def __init__(self, n_classes: int):
#         super().__init__()
#         self.cnn_block = nn.Sequential(
#             nn.Conv2d(in_channels=1, 
#                       out_channels=32, 
#                       kernel_size=5, # how big is the square that's going over the image?
#                       stride=1, # default
#                       padding='valid'),# output: 96 by 96 
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2), # default stride value is same as kernel_size
#             nn.Conv2d(in_channels=32, 
#                       out_channels=64,
#                       kernel_size=5,
#                       stride=1,
#                       padding='valid'),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2), # default stride value is same as kernel_size
#             nn.Conv2d(in_channels=64, 
#                       out_channels=128,
#                       kernel_size=(22,1),
#                       stride=1,
#                       padding=(0,1)),
#             nn.BatchNorm2d(128),
#         )
#         self.gru =  nn.GRU(input_size=128,hidden_size=64,num_layers=1,batch_first=True,bidirectional=False)
#         self.classifier = nn.Linear(in_features=64, out_features=n_classes)
#         #nn.Sequential(
#             # nn.Linear(in_features=64, 
#             #           out_features=1000),
#             # GRU receives: N, L, H when batch_first=True
#             # nn.GRU(input_size=128,hidden_size=64,num_layers=1,batch_first=True,bidirectional=False),
#         #     nn.Linear(in_features=64, 
#         #               out_features=n_classes),  # 4 classes        
#         # )
    
#     def forward(self, x: torch.Tensor):
#         x = self.cnn_block(x)
#         # print(x.shape)
#         x = x.squeeze(dim=2).permute(0,2,1) # permuting from N, f, t to N, t, f
#         x,_ = self.gru(x)
#         x = F.dropout(x,p=0.5,training=self.training)
#         # print(x.shape)
#         x = self.classifier(x)[:,-1,:]
#         # print(x.shape)
#         return x

class CNN_Long_Data(nn.Module):
    def __init__(self, n_classes: int, input_size: int, input_length: int):
        super(CNN_Long_Data, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, 4, kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        output_length= int(((input_length-1*(2-1)-1)+1)/2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length = int(((output_length-1*(2-1)-1)+1)/2)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(0.5),
            # nn.Linear(8*output_length, 500),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8*output_length, n_classes),
        )
        # self.fc2 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(128, n_classes)
        # )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.relu(self.fc1(x))
        x = self.fc1(x.squeeze(dim=1))
        return x

class CNN_Tiny(nn.Module):
    def __init__(self, n_classes: int, input_size: int, input_length: int):
        super(CNN_Tiny, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, 4, kernel_size=3, padding=0, stride=1),
            # nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length= int(((input_length-1*(5-1)-1)+1)/2)
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(4, 4, kernel_size=2, padding=0, stride=1),
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(1,2),
        #                  stride=(1,2)),
        # )
        # output_length = int(((output_length-1*(2-1)-1)+1)/2)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(0.5),
            # nn.Linear(8*output_length, 500),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4*output_length, n_classes),
        )
        # self.fc2 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(128, n_classes)
        # )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.relu(self.fc1(x))
        x = self.fc1(x.squeeze(dim=1))
        return x

class CNN_Long_Data2(nn.Module):
    def __init__(self, n_classes: int, input_size: int, input_length: int):
        super(CNN_Long_Data2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, 16, kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        output_length= int(((input_length-1*(2-1)-1)+1)/2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length = int(((output_length-1*(2-1)-1)+1)/2)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32*output_length, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )
        # self.fc2 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(128, n_classes)
        # )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.relu(self.fc1(x))
        x = self.fc1(x.squeeze(dim=1))
        return x

# Requieres specific length of 6094
class CNN_Long_Data4(nn.Module):
    def __init__(self, n_classes: int, input_size: int, input_length: int):
        super(CNN_Long_Data4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, 8, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length= int(((input_length-1*(3-1)-1)+1)/2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(1,2),
            #              stride=(1,2)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(1,2),
            #              stride=(1,2)),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length = int(((output_length-1*(3-1)-1)+1)/2)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16*output_length, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )
        # self.fc2 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(128, n_classes)
        # )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        # print(x1.shape)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = torch.cat([x1, x3], dim=1)
        x = self.conv4(x4)
        # print(x.shape)
        # x = self.relu(self.fc1(x))
        x = self.fc1(x.squeeze(dim=1))
        return x

class CNN_Long_Data3(nn.Module):
    def __init__(self, n_classes: int, input_size: int, input_length: int):
        super(CNN_Long_Data3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, 16, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length= int(((input_length-1*(3-1)-1)+1)/2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(1,2),
            #              stride=(1,2)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(1,2),
            #              stride=(1,2)),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length = int(((output_length-1*(3-1)-1)+1)/2)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32*output_length, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )
        # self.fc2 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(128, n_classes)
        # )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        # print(x1.shape)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = torch.cat([x1, x3], dim=1)
        x = self.conv4(x4)
        # print(x.shape)
        # x = self.relu(self.fc1(x))
        x = self.fc1(x.squeeze(dim=1))
        return x

class CNN_Long_Data5(nn.Module):
    def __init__(self, n_classes: int, input_size: int, input_length: int):
        super(CNN_Long_Data5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length= int(((input_length-1*(3-1)-1)+1)/2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(1,2),
            #              stride=(1,2)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(1,2),
            #              stride=(1,2)),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length = int(((output_length-1*(3-1)-1)+1)/2)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64*output_length, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )
        # self.fc2 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(128, n_classes)
        # )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        # print(x1.shape)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = torch.cat([x1, x3], dim=1)
        x = self.conv4(x4)
        # print(x.shape)
        # x = self.relu(self.fc1(x))
        x = self.fc1(x.squeeze(dim=1))
        return x

class CNN_RNN_Long_Data1(nn.Module):
    def __init__(self, n_classes: int, input_size: int, input_length: int):
        super(CNN_RNN_Long_Data1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, 16, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length= int(((input_length-1*(3-1)-1)+1)/2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length = int(((output_length-1*(3-1)-1)+1)/2)

        self.gru =  nn.LSTM(input_size=64,hidden_size=128,num_layers=3,batch_first=True,bidirectional=False, dropout=0.5)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        # print(x1.shape)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = torch.cat([x1, x3], dim=1)
        x5 = self.conv4(x4)
        x6 = self.conv5(x5)
        x = torch.cat([x4, x6], dim=1)
        x = self.conv6(x)
        x = x.squeeze(dim=2).permute(0,2,1) # permuting from N, f, t to N, t, f
        x,_ = self.gru(x)
        # print(x.shape)
        x = self.classifier(x)[:,-1,:]
        return x

class CNN_RNN_Long_Data2(nn.Module):
    def __init__(self, n_classes: int, input_size: int, input_length: int):
        super(CNN_RNN_Long_Data2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, 8, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length= int(((input_length-1*(3-1)-1)+1)/2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length = int(((output_length-1*(3-1)-1)+1)/2)

        self.gru =  nn.LSTM(input_size=32,hidden_size=64,num_layers=3,batch_first=True,bidirectional=False, dropout=0.5)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        # print(x1.shape)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = torch.cat([x1, x3], dim=1)
        x5 = self.conv4(x4)
        x6 = self.conv5(x5)
        x = torch.cat([x4, x6], dim=1)
        x = self.conv6(x)
        x = x.squeeze(dim=2).permute(0,2,1) # permuting from N, f, t to N, t, f
        x,_ = self.gru(x)
        # print(x.shape)
        x = self.classifier(x)[:,-1,:]
        return x

class CNN_RNN_Long_Data3(nn.Module):
    def __init__(self, n_classes: int, input_size: int, input_length: int):
        super(CNN_RNN_Long_Data3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length= int(((input_length-1*(3-1)-1)+1)/2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),
                         stride=(1,2)),
        )
        output_length = int(((output_length-1*(3-1)-1)+1)/2)

        self.gru =  nn.LSTM(input_size=128,hidden_size=256,num_layers=3,batch_first=True,bidirectional=False, dropout=0.5)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        # print(x1.shape)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = torch.cat([x1, x3], dim=1)
        x5 = self.conv4(x4)
        x6 = self.conv5(x5)
        x = torch.cat([x4, x6], dim=1)
        x = self.conv6(x)
        x = x.squeeze(dim=2).permute(0,2,1) # permuting from N, f, t to N, t, f
        x,_ = self.gru(x)
        # print(x.shape)
        x = self.classifier(x)[:,-1,:]
        return x