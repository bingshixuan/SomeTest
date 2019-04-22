'''
This is version 5/
try to change net structure
'''


import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import matthews_corrcoef
import sklearn.metrics as metrics

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#learning_rate = 1e-2
learning_rate = 0.002

#  1 try . This is a good network could predict result,seems right.

'''
class Cnn(nn.Module):
    def __init__(self, in_dim, out):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 200, kernel_size=7, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(200, 50, kernel_size=5, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)) # feature=4

        self.fc = nn.Sequential(
            nn.Linear(50*4 , 100),
            nn.Linear(100, 50),
            nn.Linear(50, 1))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
'''

# This is second try , better than above model mcc=0.32

'''
class Cnn(nn.Module):
    def __init__(self, in_dim, out):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(in_dim, 400, kernel_size=5, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(400, 200, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(200, 100, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)) # feature =1
        self.fc = nn.Sequential(
                nn.Linear(300, 1))
        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out  
'''


# This is the thrid try , worse than above two

'''
class Cnn(nn.Module):
    def __init__(self, in_dim, out):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(in_dim, 800, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(800, 400, kernel_size=3, padding=0, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(400, 200, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2))
        
        self.fc = nn.Sequential(
                nn.Linear(3*200, 100),
                nn.Linear(100, 1))
        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
   ''' 


# This is the forth try

'''
class Cnn(nn.Module):
    def __init__(self, in_dim, out):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(in_dim, 400, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(400, 400,  kernel_size=3, padding=0, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(400, 400, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(400, 200, kernel_size=3, padding=0, stride=1),
                nn.ReLU()
                )
        self.fc = nn.Sequential(
                nn.Linear(200, 100),
                nn.Linear(100,1)
                )
    
    def forward(self, x):
        out =self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
'''

# This is the fifth

'''
class Cnn(nn.Module):
    def __init__(self, in_dim, out):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(in_dim, 512, kernel_size=5, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(512, 1024, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(1024, 128, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)) # feature =1
        self.fc = nn.Sequential(
                nn.Linear(128*3, 1))
        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out  
'''

# This is the sixth


'''
class Cnn(nn.Module):
    def __init__(self, in_dim, out):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(in_dim, 512, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(512, 1024,  kernel_size=3, padding=0, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(1024, 1024, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(1024, 1024, kernel_size=3, padding=0, stride=1),
                nn.ReLU()
                )
        self.fc = nn.Sequential(
                nn.Linear(1024, 1)) # seventh is change 256 to 1024
    
    def forward(self, x):
        out =self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

'''

# This is the eighth

'''

class Cnn(nn.Module):
    def __init__(self, in_dim, out):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(in_dim, 512, kernel_size=3, padding=2, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(512, 1024,  kernel_size=3, padding=2, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(1024, 1024, kernel_size=3, padding=2, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(1024, 1024, kernel_size=3, padding=2, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(1024, 512, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                )
        self.fc = nn.Sequential(
                nn.Linear(512, 1))
    
    def forward(self, x):
        out =self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
'''


class Cnn_Complex(nn.Module):
    def __init__(self, in_dim, out):
        super(Cnn_Complex, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=0, stride=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, x):
        tmp_x1 = x[0]
        tmp_x2 = [x[ind] for ind in range(1, 31, 3)]
        tmp_x3 = [x[ind+1] for ind in range(1, 31, 3)]
        tmp_x4 = [x[ind+2] for ind in range(1, 31, 3)]

        tmp_x1 = self.conv1(tmp_x1)
        tmp_x2 = self.conv2(tmp_x2)
        tmp_x3 = self.conv2(tmp_x3)
        tmp_x4 = self.conv2(tmp_x4)

        out = 

        tmp_out = self.conv1()
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



model = Cnn(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
epoch=1

#This is for all svminput data as input,

allFeature = []
allObj = []

srcfile='5.5/ELSCinput.txt'
print('model is : ' + srcfile)
traindataName=srcfile
f = open(traindataName)
lines = f.readlines()
f.close()
for line in lines:
    y = line.split(' ',1)[0]
    tmpx = line.split(' ',1)[1]
    linex = tmpx.split(' ',30)
    xAll = []
    for x in linex:
        xfeature = x.split(':',1)[1]
        xAll.append(float(xfeature))
    allFeature.append(xAll)
    allObj.append(float(y))


for m in range(epoch):
    for i in range(len(allObj)):
        tmp = np.array(allFeature[i])
        tmp = tmp.reshape([1, 1, len(tmp)])
        x_train = torch.FloatTensor(tmp)
        y_train = torch.FloatTensor(np.array(allObj[i]).reshape(1,1))
        x_train = Variable(x_train)
        y_train = Variable(y_train)
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out ,y_train)
        loss.backward()
        optimizer.step()
        

res = []
for i in range(len(allObj)):
    tmp = np.array(allFeature[i])
    tmp = tmp.reshape([1, 1, len(tmp)])
    x_train = torch.FloatTensor(tmp)
    y_train = torch.FloatTensor(np.array(allObj[i]).reshape(1,1))
    x_train = Variable(x_train)
    y_train = Variable(y_train)
            
    out = model(x_train)
    predicted = np.int64(out.cpu().data.numpy()[0,0]>0.5)
    res.append(predicted)
    
#print(sum(allObj))
#print(sum(res))
#print(len(res))
#print(sum(np.array(res)==allObj)/len(allObj))
print(metrics.accuracy_score(allObj, res))
a, b, c, d = metrics.confusion_matrix(allObj, res).ravel()
print(a)
print(b)
print(c)
print(d)
#print(metrics.confusion_matrix(allObj, res).ravel()) # tn, fp, fn, tp
print(metrics.precision_score(allObj, res))
print(metrics.recall_score(allObj, res))
print(metrics.f1_score(allObj, res))
print(matthews_corrcoef(allObj, res))


rootdir = 'test_data/feature'
list = os.listdir(rootdir)
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if path.split('.',1)[1] == 'label':
        continue

    f = open(path)
    lines = f.readlines()
    f.close()
    testFeature = []
    testObj = []
    for line in lines:
        y = line.split(' ',1)[0]
        tmpx = line.split(' ',1)[1]
        linex = tmpx.split(' ',30)
        xAll = []
        for x in linex:
            xfeature = x.split(':',1)[1]
            xAll.append(float(xfeature))
        testFeature.append(xAll)
        testObj.append(float(y))

    f = open(path.split('.',1)[0]+'.label')
    lines = f.readlines()
    f.close()
    testObj = []

    for line in lines:
        y = line.split()[3]
        testObj.append(float(y))        

    res = []
    for i in range(len(testObj)):
        tmp = np.array(testFeature[i])
        tmp = tmp.reshape([1, 1, len(tmp)])
        x_test = torch.FloatTensor(tmp)
        y_test = torch.FloatTensor(np.array(testObj[i]).reshape(1,1))
        x_test = Variable(x_test)
        y_test = Variable(y_test)
        out = model(x_test)
        predicted = np.int64(out.data.numpy()[0,0]>0.5)
        res.append(predicted)

    #print(sum(res))
    #print(len(res))
    #print(sum(np.array(res)==testObj)/len(testObj))
    print(metrics.accuracy_score(testObj, res))
    a, b, c, d = metrics.confusion_matrix(testObj, res).ravel()
    print(a)
    print(b)
    print(c)
    print(d)
    #print(metrics.confusion_matrix(testObj, res).ravel())
    print(metrics.precision_score(testObj, res))
    print(metrics.recall_score(testObj, res))
    print(metrics.f1_score(testObj, res))
    print(matthews_corrcoef(testObj, res))
