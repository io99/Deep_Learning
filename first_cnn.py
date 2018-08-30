import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#device configuration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#declaring the hyper_parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

#MNIST dataset, transforming into tensor

train_dataset = torchvision.datasets.MNIST(root='../../data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            )
test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                            train=False,
                                            transform=transforms.ToTensor())

#Data 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
#what is shuffle?
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False)

#cnn
class Covnet(nn.Module):
    def _init_(self, num_classes=10):
        super(Covnet,self)._init_() #what are we doing here?
        self.layer1 =nn.Sequential(
            nn.Conv2d(1,16,kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5,stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(7*7*32, num_classes)))


