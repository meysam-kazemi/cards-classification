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


# Build model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.convs2 = nn.Conv2d(6, 6, 5, stride=(2,2))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.convs2 = nn.Conv2d(16, 16, 5, stride=(2,2))
        self.fc1 = nn.Linear(16*53*53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 53)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train Loop
for epoch in range(20):  # loop over the dataset multiple times
  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
  print(f'[{epoch + 1} |  loss: {running_loss / 2000:.3f}')
  running_loss = 0.0

print('Finished Training')

