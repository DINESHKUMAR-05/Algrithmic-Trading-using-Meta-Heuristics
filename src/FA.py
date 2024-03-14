from torch import optim
import numpy as np
import gc
import random
import os
from src.genetic_algorithm import seed_everything
from src.models import *
from fireflyalgorithm import FireflyAlgorithm
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

class FireFlyAlg:
  def __init__(self, device, criterion, X_val, y_val, model, pop_size=20, alpha=1.0, betamin=1.0, gamma=0.01, max_iter=10, seed=None):
    self.pop_size=pop_size
    self.alpha=alpha
    self.betamin=betamin 
    self.gamma=gamma
    self.seed=seed
    self.model=model
    self.device=device
    self.criterion = criterion
    self.X_val = X_val
    self.y_val = y_val
    self.max_iter=max_iter

  def objFn(self,params):
    lr = params[0]
    epoch = int(params[1])
    hidden_units = int(params[2])
    num_layers = int(params[3])
    if self.model == 'LSTM':
      seed_everything(77)
      Model = LSTM(input_size=self.X_val.shape[2],
              hidden_size=hidden_units,
              num_layers=num_layers).to(self.device)
      
    elif self.model == 'GRU':
      seed_everything(77)
      Model = GRU(input_size=self.X_val.shape[2],
              hidden_size=hidden_units,
              num_layers=num_layers).to(self.device)
        
    else:
      raise ValueError('Only LSTM and GRU blocks are available for optimization.')


    optimizer = optim.Adam(Model.parameters(), lr=lr)
    seed_everything(77)
    train(Model, self.criterion, optimizer, self.device, self.X_val, self.y_val, epoch, 
          verbose=False, return_loss_history=False, compute_test_loss=False)
          
    return predict(Model, self.X_val, self.y_val, self.criterion, self.device)
  
  def fit(self, lb, ub, dim=10):
    fa= FireflyAlgorithm(pop_size=self.pop_size, alpha=self.alpha, betamin=self.betamin, gamma=self.gamma, seed=None)
    return fa.run(function=objFn, dim=dim, lb=lb, ub=ub, max_evals=self.max_iter)
