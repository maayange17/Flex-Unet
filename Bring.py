import torch.nn as nn
import torch

def BringAct(ACT_KIND):
  if ACT_KIND == '"ReLU"':
    Act = nn.ReLU()
  return Act

def BringNorm(NORM_KIND):
  if NORM_KIND == '"Batch"':
    NORM = nn.BatchNorm2d
  return NORM

def BringPool(POOL_KIND,POOL_SIZE,STRIDE_SIZE):
  if POOL_KIND == '"Max"':
    pool = nn.MaxPool2d(POOL_SIZE,STRIDE_SIZE)
  return pool

def BringLoss(Loss):
  if Loss == '"L1"':
    loss = nn.L1Loss()
  if Loss == '"L2"':
    loss = nn.MSELoss()
  return loss

def BringOpt(Opt):
  if Opt == '"Adam"':
    opt = torch.optim.Adam
  return opt