"""
this program has some bug i dont know how to solve
"""
from torch.autograd import Variable

from Logistic_Regression.model import *
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
with open('data.txt', 'r') as f:
    data_list = f.readlines()
    data_list = [i.split('\n')[0] for i in data_list]
    data_list = [i.split(',') for i in data_list]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]
    dataset=data

x0 = list(filter(lambda x: x[-1] == 0.0, data))
x1 = list(filter(lambda x: x[-1] == 1.0, data))
data0_0 = [i[0] for i in x0]
data0_1 = [i[1] for i in x0]
data1_0 = [i[0] for i in x1]
data1_1 = [i[1] for i in x1]



data=random.shuffle(dataset)
y=[i[2] for i in dataset]
x0=[i[0] for i in dataset]
x1=[i[1] for i in dataset]
x0=torch.from_numpy(np.asarray(x0,dtype=float))
x1=torch.from_numpy(np.asarray(x1,dtype=float))
x=torch.cat((x0.unsqueeze(1),x1.unsqueeze(1)),1)
y=torch.from_numpy(np.asarray(y,dtype=float))
cuda=False
if torch.cuda.is_available():
    cuda=True
model=LogisticRegression()
if cuda:
    model.cuda()
criterion=nn.BCELoss()
optimizer=optim.SGD(model.parameters(),lr=1e-3)
for epoch in range(5000):
    if cuda:
        x=Variable(x).cuda()
        y=Variable(y).cuda()
    else:
        x = Variable(x)
        y = Variable(y)
    x=x.type(torch.float32)
    y=y.type(torch.float32)
    out=model(x)
    loss = criterion(out, y)
    mask=out.ge(0.5).float()
    correct=(mask==y).sum()
    acc=correct.data/x.size(0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%1000==0:
        print("epoch:{} loss:{:.6f} acc:{:.6f}".format(epoch+1,loss.data,acc))

w0,w1=model.lr.weight[0]
w0=w0.data
w1=w1.data
b=model.lr.bias.data[0]
plot_x=np.arange(30,100,0.1)
plot_y=(-w0*plot_x-b)/w1
plt.plot(plot_x,plot_y)
plt.show()
