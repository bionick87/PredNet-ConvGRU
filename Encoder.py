import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


class Encoder(nn.Module):

    def __init__(self, input_state_size,\
                 hidden_size, kenerl_size, image_size):

        super(Encoder, self).__init__()
         
        self.num_f_input  = input_state_size[0]
        self.num_f_state  = input_state_size[1]
        self.kenerl_size  = kenerl_size
        self.hidden       = hidden_size
        self.image_size   = image_size
        self.Conv_Alt     = nn.Conv2d(self.num_f_input, self.hidden,\
                                      self.kenerl_size, padding=self.kenerl_size//2)
        self.Conv_Alt_cap = nn.Conv2d(self.num_f_state, self.hidden,\
                                      self.kenerl_size, padding=self.kenerl_size//2)
    def forward(self, x_t, R_lt, first):
        # ==> Equation (1) page 3 PredNet paper
        A_lt = None
        if first is True: 
          A_lt   = x_t 
        else: 
          A_lt   = f.max_pool2d(f.relu(self.Conv_Alt(x_t)), 2, 2)
        # ==> Equation (2) page 3 PredNet paper 
        A_lt_cap = f.relu(self.Conv_Alt_cap(R_lt))
        # ==> Equation (4) page 3 PredNet paper (dimension of stack 1)
        E_lt     = torch.cat((f.relu(A_lt - A_lt_cap),f.relu(A_lt_cap - A_lt)),1)
        return E_lt
      
def test(channel, x_lt, R_o, kernel_size):
    encode_layer1  = Encoder([channel,3], 3, kernel_size, True)
    E_o            = encode_layer1(x_lt,R_o)
    # Fake Encoding:
    R_l1           = Variable(torch.randn(1, 32, 128, 128)) 
    encode_layer2  = Encoder([6,32], 32, kernel_size, False)
    R_l2           = encode_layer2(E_o,R_l1)

if __name__ == '__main__':
   channel_image = 3 
   kernel_size   = 3
   x_lt          = Variable(torch.randn(1, 3, 256, 256))
   R_lt          = Variable(torch.randn(1, 3, 256, 256))
   test (channel_image, x_lt, R_lt, kernel_size)

    
