#import library
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.opitm as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter(action='ignore')
device='cuda' if torch.cuda.is_available() else 'cpu'

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP ,self).__init__()
        self.linear1=nn.Linear(input_size, 8)
        self.linear2=nn.Linear(8, 4)
        self.linear3=nn.Linear(4, 1)
        self.dropout=nn.Dropout(p=0.2)
    def forward(self, x):
        x=self.linear1(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.Linear2(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.linear3(x)
        return x

#parameter define
n_lags=3
valid_size=12
batch_size=10
n_epochs=1000

#set seed, loss function, optimizer
torch.manual_seed(42)
MLP=MLP(n_lags).to(device)
loss_function=nn.MSELoss()
optimizer=optim.Adam(MLP.parameters(), lr=0.0001)
#check the model structure
MLP

#transform raw data into MLP acceptable inputs
def create_input_data(series, n_lags=1):
    X, y=[], []
    for step in range(len(series)-n_lags):
        end_step=step+n_lags
        X.append(series[step:end_step])
        y.append(series[end_step])
    return np.array(X), np.array(y)

#pass the parameters and raw data into the function we defined and create tensors
X, y=create_input_data(raw_data, n_lags)
X_tensor=torch.from_numpy(X).float()
y_tensor=torch.from_numpy(y).float().unsqueeze(dim=1)

#create training and validation datasets
dataset=TensorDataset(X_tensor, y_tensor)

valid_index=len(X)-valid_size 
train_dataset=Subset(dataset, list(range(valid_index)))
valid_dataset=Subset(dataset, list(range(valid_index, len(X))))

train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size)
valid_loader=DataLoader(dataset=valid_dataset, batch_size=batch_size)

#model training
print_every=20
train_losses, valid_losses=[],[]

for epoch in range(n_epochs):
  running_loss_train = 0
  running_loss_valid = 0

  MLP.train()

  for x_batch, y_batch in train_loader: 
    optimizer.zero_grad() 
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    y_hat = MLP(x_batch) #obtain the predictions
    loss = loss_function(y_batch, y_hat)
    loss.backward() #backward propagation
    optimizer.step() #update the weights
    running_loss_train += loss.item()*x_batch.size(0) 
  epoch_loss_train=running_loss_train / len(train_loader.dataset) 
  train_losses.append(epoch_loss_train)

  with torch.no_grad(): 
    MLP.eval() 

    for x_valid, y_valid in valid_loader:
      x_valid=x_valid.to(device)
      y_valid=y_valid.to(device)
      y_hat=MLP(x_valid)
      loss=loss_function(y_valid, y_hat)
      running_loss_valid += loss.item()*x_valid.size(0)
    epoch_loss_valid = running_loss_valid / len(valid_loader.dataset)

    if epoch>0 and epoch_loss_valid < min(valid_losses):
      best_epoch = epoch
      torch.save(MLP.state_dict(), './mlp.pth')

    valid_losses.append(epoch_loss_valid)
  
  if epoch % print_every==0:
    print(f"<{epoch}> – Train. loss: {epoch_loss_train:.2f} \t Valid. loss: {epoch_loss_valid:.2f}")
print(f'Lowest loss recorded in epoch: {best_epoch}')

#make prediction based on validation dataset
y_pred, y_valid=[], []
with torch.no_grad():
  MLP.eval()
  for x_val, y_val in valid_loader:
    x_valid = x_val.to(device) 
    y_pred.append(MLP(x_valid))
    y_valid.append(y_val)
y_pred=torch.cat(y_pred).numpy().flatten() #convert tensor to numpy array
y_valid=torch.cat(y_valid).numpy().flatten()

#prediction evaluation
mlp_mse = mean_squared_error(y_valid, y_pred)
mlp_rmse = np.sqrt(mlp_mse)
print(f"MLP's Forecast – MSE: {mlp_mse:.2f}, RMSE: {mlp_rmse:.2f}")



## Another way to build MLP using sklearn 
from sklearn.neural_network import MLPRegressor, MLPClassifier
mlp=MLPRegressor( 
    hidden_layer_sizes=(8, 4, ),
    learning_rate='constant',
    batch_size=5,
    max_iter=1000,
    random_state=42)

#data split
valid_i=len(X)-valid_size
X_train = X[:valid_i, ]
y_train = y[:valid_i]
X_valid = X[valid_i:, ]
y_valid = y[valid_i:]

#fit model and make prediction
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_valid)

#prediction evaluation
sklearn_mlp_mse = mean_squared_error(y_valid, y_pred)
sklearn_mlp_rmse = np.sqrt(sklearn_mlp_mse)
print(f"Scikit-Learn MLP's forecast - MSE: {sklearn_mlp_mse:.2f}, RMSE: {sklearn_mlp_rmse:.2f}")