
###################################################
# Nicolo Savioli, 2017 -- Conv-GRU pytorch v 1.0  #
###################################################

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable

from ConvGRUCell import ConvGRUCell


class Decoder(nn.Module):

    def __init__(self, Elt_size, Rlt_top_size,\
                 hidden_size, kernel_size):
        super(Decoder, self).__init__()
        self.ConvGRU_error    = 0
        self.Elt_size         = Elt_size
        self.Rlt_top_size     = Rlt_top_size
        self.hidden_size      = hidden_size
        self.kernel_size      = kernel_size
        self.GRUcell          = ConvGRUCell(self.Elt_size + self.Rlt_top_size,\
                                            self.hidden_size,self.kernel_size)
        
    def forward(self, Elt, Rlt_top, Rlt_state):
        if Rlt_top is None: 
            self.ConvGRU_error  = self.GRUcell(Elt,Rlt_state)
        else:
            up_Rlt_top          = f.upsample(Rlt_top, scale_factor=2)
            tot_error_in_GRU    = torch.cat((Elt,up_Rlt_top),1)
            self.ConvGRU_error  = self.GRUcell(tot_error_in_GRU,Rlt_state)
        return self.ConvGRU_error

def test():
    Elt_size          = 32 
    Rlt_top_size      = 0 
    hidden_size       = 32 
    kernel_size       = 3 
    image_size        = 256
    cuda_flag         = True

    Decode            = Decoder(Elt_size, Rlt_top_size, hidden_size,\
                        image_size, kernel_size)

    # Decode start to create first image:
    Elt       = Variable(torch.randn(1, Elt_size, image_size, image_size))
    GRU_state = Variable(torch.zeros(1, Elt_size, image_size, image_size))

    if cuda_flag == True:
       Elt       = Elt.cuda()
       #Decode    = Decode.cuda()
       #GRU_state = GRU_state.cuda()
       Decode    = Decode
       GRU_state = GRU_state
    
    err = Decode(Elt,None,GRU_state) 
    print(err.data.size())


if __name__ == '__main__':
   test()








       




         
