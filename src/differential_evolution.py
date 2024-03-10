from torch import optim
import numpy as np
import gc
import random
import os
from src.genetic_algorithm import seed_everything
from src.models import *
from scipy.optimize import differential_evolution
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

class DE:
  def __init__(self, device, criterion, X_val, y_val, model, popsize: int = 100, mutation: tuple = (0.5,1),updating : str = "immediate", recombination: float = 0.7,  maxiter: int = 100):
    self.popsize= popsize
    self.mutation = mutation
    self.recombination = recombination
    self.maxiter = maxiter
    self.model=model
    self.updating= updating
    self.device=device
    self.criterion = criterion
    self.X_val = X_val
    self.y_val = y_val

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

  def fit(bounds):
    return differential_evolution(func=objFn,bounds=bounds,maxiter=self.maxiter,updating=self.updating,popsize=self.popsize,mutation=self.mutation,recombination=self.recombination,disp=True,seed=77)


