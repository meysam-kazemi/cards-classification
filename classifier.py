import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# Configuration variables
TRAIN_DATA_PATH = "./cards/train/"
TEST_DATA_PATH = "./cards/test/"
VALID_DATA_PATH = "./cards/valid/"
DATAFRAME_PATH = "./cards/cards.csv"
BATCH_SIZE = 64

TRANSFORM_IMG = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
])

# Load Data
train = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_loader  = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid = torchvision.datasets.ImageFolder(VALID_DATA_PATH, transform=TRANSFORM_IMG)
valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Classes
df = pd.read_csv(DATAFRAME_PATH)
classes = []
for i in range(53):
  classes.append((df[df['class index']==i]['labels']).iloc[0])

# Save Classes
with open("classes.txt", "w") as f:
  for c in classes:
    f.write(c+"\n")

# Set Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\033[33m"+"device".center(25,"-")+"\033[00m")
print(device.center(25))


# Build model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk1 = self.block(3,6,9,5)
        self.blk2 = self.block(9,12,18,3)
        # self.blk3 = self.block(18,20,22,3)
        self.fc1 = nn.Linear(1800,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 53)
        self.act = F.relu

    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)

        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        return x

    def block(self, input, latent, output, kernel_size):
      blk = nn.Sequential(
          nn.Conv2d(input, latent, kernel_size),
          nn.Conv2d(latent, latent, kernel_size, stride=2),
          nn.Conv2d(latent, output, kernel_size),
          nn.Conv2d(output, output, kernel_size, stride=2)
      )
      return blk
          
net = Net().to(device)

# Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train Loop
for epoch in range(500):  # loop over the dataset multiple times
  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
  print(f'epoch : {epoch + 1} - loss: {running_loss / 2000:.3f}')
  running_loss = 0.0

print('Finished Training')

