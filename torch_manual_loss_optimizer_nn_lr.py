''' 
  1) Design: input, output size, forward pass
  2) loss and optimizer
  3) Training loop:
      -forward compute prediction (and loss)
      -backward (grads)
      -update weights aka parameters
'''


# from https://github.com/python-engineer/pytorchTutorial/blob/master/05_1_gradientdescent_manually.py

import torch
#import torch.nn as nn

# Compute every step manually

# Linear regression
# f = w * x 

# here : f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
#model=nn.Linear

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model output
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

# Torch has standard gradients already implemented
# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
#def gradient(x, y, y_pred):
#    return np.dot(2*x, y_pred - y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero grads
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
     
print(f'Prediction after training: f(5) = {forward(5):.3f}')
