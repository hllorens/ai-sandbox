#/usr/bin/python3

import torch
# from https://github.com/python-engineer/pytorchTutorial/blob/master/05_1_gradientdescent_manually.py
'''
  1) Design: input, output size, forward pass
  2) define loss and optimizer
  3) Training loop:
      -forward compute prediction (and loss)
      -backward (grads)
      -update weights aka parameters
'''

# Linear regression
# f = w * x + b
# here : f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5],dtype=torch.float32)
print(X)
print(X_test)
print(X.shape)
n_samples,n_features=X.shape
print(n_samples,n_features)
input_size=n_features
output_size=n_features

class LinRegModel(torch.nn.Module):
    def __init__(self):
        super(LinRegModel,self).__init__()
        self.layer1=torch.nn.Linear(1,1)

    def forward(self, x):
        x=self.layer1(x)
        return x


model=LinRegModel()
print(model)
output=model(X)
print(f"input:{X}")
print(f"output:{output}")
print(f"expect:{Y}")


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 300
loss=torch.nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(n_iters):
    # predict = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    #with torch.no_grad():
    #    w -= learning_rate * w.grad
    optimizer.step()

    # zero grads
    #w.grad.zero_()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters() # unpack parameters
        print(f'epoch {epoch+1}: w (model.params) = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
