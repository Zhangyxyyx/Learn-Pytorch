from torch.autograd import Variable
import numpy as np
from one_dimensional_regression.model import *
import torch.optim as optim
import matplotlib.pyplot as plt

model = LinearRegression()
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042], [10.791], [5.313], [7.997],
                    [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596], [2.53], [1.221],
                    [2.827], [3.456], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train=torch.from_numpy(x_train)
y_train=torch.from_numpy(y_train)

epoch = 10000
for i in range(epoch):
    inputs = Variable(x_train)
    target = Variable(y_train)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        target = target.cuda()
    out = model(inputs)
    loss = criterion(out, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1)%10==0:
        print("Epoch[{}/{}],loss:{:.6f}".format(i+1,epoch,loss.data))


model.eval()
model.cpu()
predict=model(Variable(x_train))
predict=predict.data.numpy()
plt.plot(x_train.numpy(),y_train.numpy(),'ro',label='Original data')
plt.plot(x_train.numpy(),predict,label='Fitting Line')
plt.show()