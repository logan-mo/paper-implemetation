import os
import cv2                              # For readig and writing images
import glob                             # For getting list of all files and folders in a directory
import torch                            # For training neural networks etc
import torchvision                   # For some basic mathematical operations
import pandas as pd                     # For storing metadata in tables
from torchvision.models import mobilenet_v2

class ALL_IDB2(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform1=None,transform2=None ):
        image_paths = glob.glob(os.path.join(folder_path, '*.tif'))
        self.data = pd.DataFrame({
            'image_paths':  image_paths, 
            'labels': [x[-5] for x in image_paths]
            })
        
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_id = self.data.iloc[idx, :]
        img = cv2.imread(img_path)
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = int(class_id)
        if not self.transform1 is None:
            img_tensor = self.transform1(img_tensor)
            # print(img_tensor.dtype)
            img_tensor = torch.as_tensor(img_tensor, dtype=torch.uint8)
            
            img_tensor = self.transform2(img_tensor)
            
            img_tensor = torch.as_tensor(img_tensor, dtype=torch.float32)
        return img_tensor, torch.tensor(class_id)
    
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mbv2 = mobilenet_v2(pretrained=True)
        self.mbv2.classifier[1] = torch.nn.Linear(self.mbv2.classifier[1].in_features, 1)
        torch.set_grad_enabled(True)
        self.sigm=torch.nn.Sigmoid()
       
    def forward(self, x):
        x=self.mbv2(x)
        x=self.sigm(x)
        return x
    
batch_size = 8
learning_rate = 0.001

transforms1 = torchvision.transforms.Compose(
[
    torchvision.transforms.Resize((512, 512)), 
])
transforms2 = torchvision.transforms.Compose(
[
    torchvision.transforms.RandomEqualize(1)
])

dataset = ALL_IDB2(os.path.join('data', 'ALL_IDB2', 't_aug'), transform1=transforms1, transform2=transforms2)
# torchvision.transforms.functional.convert_image_dtype(dataset, 'unint8')
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])  

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

images, labels = next(iter(train_dataloader)) 
print("images-size:", images.shape) 
print("out-size:", labels.shape)        # [4]

out = torchvision.utils.make_grid(images)           # Make a grid of images

net = Model()
net = net.to(device)

classes = [0,1]
lr = 0.01
n_epochs = 20

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


#if device == 'cuda:0':
#    net = torch.nn.DataParallel(net)
#    cudnn.benchmark = True

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = torch.zeros((len(train_dataloader) ))
    correct = 0
    total = 0
    # loop = tqdm(train_dataloader)
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs=outputs.flatten()
        
        loss = criterion(outputs.squeeze(), targets.float())
        
        loss.backward()
        optimizer.step()

        train_loss[batch_idx] = loss.item()
        predicted = outputs.ge(0.5).float() # .ge = greater tha or equals
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # print("input", targets)
        # print("outputs:", outputs)
        # print('correct:', correct)
        # print('total', total)
 
        # print(targets.size(0), predicted.eq(targets).sum().item())

        # loop.set_description(f"Epoch [{epoch}/{epoch}]")
        # loop.set_postfix(loss=torch.rand(1).item(), acc=torch.rand(1).item())
        print(f"idx:{batch_idx}, batches:{(batch_idx+1)}/{len(train_dataloader)}, accuracy:{100.*correct/total}")

        # progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss.mean().item()

def test(epoch,model_name):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs=outputs.flatten()
            loss = criterion(outputs.squeeze(), targets.float())

            test_loss += loss.item()
            predicted = outputs.ge(0.5).float()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+model_name+'.pth')
        best_acc = acc
        
    print(f"idx:{batch_idx}, batches:{batch_idx+1}/{len(test_dataloader)}, accuracy:{acc}")
    return test_loss

epoch = 0
n_epochs = 20
while(epoch < n_epochs):
    train_loss = train(epoch) / len(train_dataloader)
    validation_loss = test(epoch,'final_train_allidb2') / len(test_dataloader)
    scheduler.step()
    epoch+=1