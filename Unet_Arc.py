from torchvision import models
from torchsummary import summary
import torch.nn as nn
import torch
from collections import OrderedDict
from Bring import BringAct,BringNorm,BringPool


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate,self).__init__()
    
    def forward(self,data1,data2):
      return torch.cat((data1, data2),1)
        

class BASICCONVORDECONVLAYER(nn.Module):
    def __init__(self,C_IN,C_OUT,Params,DEOREN,CONV):
      super(BASICCONVORDECONVLAYER,self).__init__()
      self.isconv = CONV
      self.act = BringAct(Params['Act'])
      self.NORM = Params['Norm_layer']
      self.NORMLayer = BringNorm(Params['Norm_layer_kind'])
      self.NORMLayer = self.NORMLayer(int(C_OUT))
      if DEOREN:
        str1 = 'encoder'
      else:
        str1 = 'decoder'
      KERNEL_SIZE = tuple(map(int, Params['Kernel_size_'+str1].split(',')))
      STRIDE_SIZE = tuple(map(int, Params['Stride_size_'+str1].split(','))) 
      KERNEL_SIZE_DEC = tuple(map(int, Params['Kernel_size_deconvlayer'].split(',')))
      self.conv = nn.Conv2d(in_channels=int(C_IN), out_channels = int(C_OUT), kernel_size = KERNEL_SIZE, padding = Params['padding_conv_layer'][1:len(Params['padding_conv_layer'])-1],stride = STRIDE_SIZE)
      self.deconv = nn.ConvTranspose2d(in_channels=int(C_IN), out_channels = int(C_OUT), kernel_size = KERNEL_SIZE_DEC,stride = KERNEL_SIZE_DEC)
      self.C_IN = C_IN
      self.C_OUT = C_OUT

    def forward(self,x):
      if self.isconv:
        out = self.conv(x)
      else:
        out = self.deconv(x)
      out = self.act(out)
      if self.NORM:
        out = self.NORMLayer(out)
      return out

class EncoderBlock(nn.Module):
  def __init__(self,number_of_level,Params,W_POOL):
    super(EncoderBlock,self).__init__()
    self.convs = OrderedDict()
    self.NUMBER_OF_CONV = int(Params['Blocks_in_level'])
    self.num_of_lev = number_of_level
    B_F_N = int(Params['Basic_filters_num'])
    D_O_F = int(Params['Duplication_of_filters'])
    Levels = int(Params['Levels'])
    B_H = int(Params['Basic_CH'])  
    if number_of_level == -1:
      C_IN = B_F_N*(D_O_F**(Levels-2))
      C_OUT = B_F_N*(D_O_F**(Levels-1))
    if number_of_level == 0:
      C_IN = B_H
      C_OUT = B_F_N
    if number_of_level > 0:
      C_IN = B_F_N*(D_O_F**(number_of_level-1))
      C_OUT = B_F_N*(D_O_F**(number_of_level))
    
    self.convs[str(0)] = BASICCONVORDECONVLAYER(C_IN,C_OUT,Params,True,True)
    for i in range(self.NUMBER_OF_CONV-1):
      self.convs[str(i+1)] =  BASICCONVORDECONVLAYER(C_OUT,C_OUT,Params,True,True)
    
    
    
    self.conv_layers = nn.Sequential(self.convs)
    self.W_POOL = W_POOL
    P_Z =  tuple(map(int,Params['Pool_size'].split(',')))
    P_S = tuple(map(int,Params['Pool_stride'].split(',')))
    self.Pool = BringPool(Params['Pool_kind'],P_Z ,P_S)
    self.DROPOUT = Params['Dropout_encoder']
    if self.DROPOUT == "True":
      self.drop = nn.Dropout2d(float(Params['Dropout']))
    
    self.C_IN = C_IN
    self.C_OUT = C_OUT
    self.filt = None
  
  def Filt(self):
    return self.filt
    
  
  def forward(self,x):
    out = self.conv_layers(x)
    if self.DROPOUT == "True":
      out = self.drop(out)
    if self.W_POOL:
      filt = torch.clone(out)
      out = self.Pool(out)
      self.filt = filt
      return out    
    return out

class DecoderBlock(nn.Module):
  def __init__(self,number_of_level,Params):
    super(DecoderBlock,self).__init__() 
    self.convs = OrderedDict()
    B_F_N = int(Params['Basic_filters_num'])
    D_O_F = int(Params['Duplication_of_filters'])
    Levels = int(Params['Levels'])
    B_H = int(Params['Basic_CH'])
    if number_of_level == 0:
      C_IN = B_F_N
      C_OUT = B_H
    else:
      C_OUT = B_F_N*(D_O_F**(number_of_level-1))
      C_IN = B_F_N*(D_O_F**(number_of_level))
    
    self.NUMBER_OF_CONV = int(Params['Blocks_in_level'])
    self.doconc =  Params['Skip']
    if self.doconc == "True":
      C_TAG = C_OUT
    else:
      C_TAG = C_IN

    self.convs[str(0)] = BASICCONVORDECONVLAYER(C_IN,C_OUT,Params,False,True)
    for i in range(self.NUMBER_OF_CONV-1):
      self.convs[str(i+1)] =  BASICCONVORDECONVLAYER(C_OUT,C_OUT,Params,False,True)
    
    self.conv_layers = nn.Sequential(self.convs)

    self.deconv = BASICCONVORDECONVLAYER(C_IN,C_TAG,Params,False,False)
    self.concat = Concatenate()
    self.DROPOUT = Params['Dropout_decoder']
    if self.DROPOUT == "True":
      self.drop = nn.Dropout2d(float(Params['Dropout']))
    self.C_IN = C_IN
    self.C_OUT = C_OUT
    self.C_TAG = C_TAG

  def data(self,data):
    self.data = data

  def forward(self,x):
    out = self.deconv(x)
    if self.doconc == "True":
      out = self.concat(out,self.data)
    out = self.conv_layers(out)
    if self.DROPOUT:
      out = self.drop(out)    
    return out
    


class Encoder(nn.Module):
  def __init__(self,Params):
    super(Encoder,self).__init__()    
    self.EncoderBlocks = OrderedDict()
    self.NUM_OF_LEVELS = int(Params['Levels'])
    self.filts = {}
    for i in range(self.NUM_OF_LEVELS-1):
      self.EncoderBlocks[str(i)] = EncoderBlock(i,Params,True)    
    
    self.encodeblocks = nn.Sequential(self.EncoderBlocks)
  
  def Filts(self):
    return self.filts  
  
  def forward(self,x):
    out = self.encodeblocks(x)
    for i in range(self.NUM_OF_LEVELS-1):
      self.filts[i] = self.EncoderBlocks[str(i)].Filt()
    return out

class Decoder(nn.Module):
  def __init__(self,Params):
    super(Decoder,self).__init__()
    self.DecoderBlocks = OrderedDict()
    self.NUM_OF_LEVELS = int(Params['Levels'])
    for i in range(self.NUM_OF_LEVELS-1):
      self.DecoderBlocks[str(i)] = DecoderBlock(self.NUM_OF_LEVELS-i-1,Params)
    self.decodeblocks = nn.Sequential(self.DecoderBlocks)
  
  def Filts(self,filts):
    self.Filts = filts

  def forward(self,x):
    for i in range(self.NUM_OF_LEVELS-1):
      self.DecoderBlocks[str(i)].data(self.Filts[len(self.Filts)- i - 1])
    out = self.decodeblocks(x)

    return out

class Bottleneck(nn.Module):
    def __init__(self,Params):
      super(Bottleneck,self).__init__()   
      self.convlayer =  EncoderBlock(-1,Params,False)
    
    def forward(self,x):
      out = self.convlayer(x)
      return out


class Unet(nn.Module):
  def __init__(self,Params):
    super(Unet,self).__init__()
    B_F_N = int(Params['Basic_filters_num'])
    B_H = int(Params['Basic_CH'])
  
    self.Params = Params
    self.encoder = Encoder(self.Params)
    self.bottleneck = Bottleneck(self.Params)
    self.decoder = Decoder(self.Params)
    self.finallayer = nn.Conv2d(in_channels = B_F_N, out_channels = B_H, kernel_size = (1,1), padding = 'same',stride = (1,1))
    self.finalact = BringAct(Params['Last_layer_act'])
    
  
  def forward(self,x):
    out = self.encoder(x)
    out = self.bottleneck(out)
    self.decoder.Filts(self.encoder.Filts())
    out = self.decoder(out)
    out = self.finallayer(out)
    out = self.finalact(out)
    return out


