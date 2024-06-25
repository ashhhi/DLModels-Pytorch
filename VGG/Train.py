import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from VGG import VGG16_Original, VGG16_Modified

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using Cuda device")
else:
    device = torch.device("cpu")
    print("MPS device not available, using CPU")

batch_size = 32

writer = SummaryWriter(r"./Logs")

# FashionMNIST = datasets.FashionMNIST(r'./Dataset', train=True, download=True, transform=ToTensor())
train_data = datasets.CIFAR10(r'/Users/shijunshen/Documents/Code/PycharmProjects/DLModels-Pytorch/Dataset', train=True, download=True, transform=ToTensor())
test_data = datasets.CIFAR10(r'/Users/shijunshen/Documents/Code/PycharmProjects/DLModels-Pytorch/Dataset', train=False, download=True, transform=ToTensor())
# print(len(train_data), len(test_data))

train_loader = DataLoader(train_data, batch_size, True)
test_loader = DataLoader(test_data, batch_size, True)

model = VGG16_Modified().to(device)
print(model)

loss_function = CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

epoch_num = 5
for epoch in range(epoch_num):
    print(f"-------------------第{epoch}轮-------------------")
    model.train()
    with tqdm(train_loader, desc="Epoch {}".format(epoch), unit="batch") as t:
        for data in t:
            img, target = data
            img = img.to(device)
            target = target.to(device)
            prediction = model(img)
            loss = loss_function(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=loss)
            writer.add_scalar('loss', loss)

    try:
        save_path = '/Users/shijunshen/Documents/Code/PycharmProjects/DLModels-Pytorch/ModelSave/VGG.pth'
        torch.save(model.state_dict(), save_path)
        print(f'save model to {save_path}')
    except Exception as e:
        print(f'save model fail')
    model.eval()
    with tqdm(test_loader, desc="Epoch {} Test".format(epoch), unit="batch") as t:
        test_total_loss = 0
        for data in t:
            img, target = data
            img = img.to(device)
            target = target.to(device)
            with torch.no_grad():
                prediction = model(img)
                loss = loss_function(prediction, target)
            test_total_loss += loss
            t.set_postfix(total_loss=test_total_loss)
    print()

