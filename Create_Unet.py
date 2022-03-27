from configparser import ConfigParser
import Unet_Arc
from torchsummary import summary
import torch



def Unet_with_params(Params,device1):
  NewUnet = Unet_Arc.Unet(Params).to(device1)
  Input_size = tuple(map(int,Params['Input_Size'].split(',')))
  summary(Unet_Arc.Unet(Params), Input_size,device = 'cpu')
  return NewUnet

def Init_Unet():
  config = ConfigParser()
  config.read('Config_Model.ini')
  Params = config['Model_Params']
  device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device1)
  return Unet_with_params(Params,device1)
