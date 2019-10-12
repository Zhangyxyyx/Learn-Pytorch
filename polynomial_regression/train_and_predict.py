from torch.autograd import Variable
import numpy as np
from polynomial_regression.model import *
import torch.optim as optim
import matplotlib.pyplot as plt


def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)

cuda = False
if torch.cuda.is_available():
    cuda = True
W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])
if cuda:
   W_target=W_target.cuda()
   b_target=b_target.cuda()



def get_batch(batch_size=32, cuda=False):
    x = torch.randn(batch_size)
    if cuda:
        x = x.cuda()
    x_matrix = make_features(x)
    y = (x_matrix.mm(W_target) + b_target)
    if cuda:
        return Variable(x).cuda(), Variable(x_matrix).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(x_matrix), Variable(y)



model = poly_regression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
epoch = 0
while True:
    batch_x, batch_x_matrix, batch_y = get_batch(128,cuda)
    if cuda:
        batch_x.cuda()
        batch_y.cuda()
        batch_x_matrix.cuda()
        model.cuda()
    out = model(batch_x_matrix)
    loss = criterion(out, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if epoch % 20 == 0:
        print("eopch: {} loss: {:.6f}".format(epoch, loss.data))
    if loss.data<1e-4:
        break

batch_x, batch_x_matrix, batch_y = get_batch(128,cuda)

model.eval()
model.cuda()
predict=model(batch_x_matrix)

batch_x=batch_x.cpu()
batch_y=batch_y.cpu().squeeze(1)
predict=predict.cpu().squeeze(1)
predict=predict.data.numpy()
plt.plot(batch_x.numpy(),batch_y.numpy(),'ro',Label="Origin data")
plt.plot(batch_x.numpy(),predict,'ro',label='fitting line',color="black")
plt.show()