import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
from Encoder import Encoder
from Decoder import Decoder
import numpy as np 
import os
import cv2 

class PrednetModel(nn.Module):

    def __init__(self,number_of_layers,\
                      image_size,numCh,\
                      SaveModelPath,\
                      OpenModelPath,\
                      numSaveIter):
        super(PrednetModel, self).__init__()
        self.image_size       = image_size
        self.input_layer_size = (3, 6,  32, 64,  128, 256, 512 )
        self.hidden_size      = (3, 16, 32, 64,  128, 256, 512 ) 
        self.Elt_size         = (6, 32, 64, 128, 256, 512      )
        self.SaveModelPath    = SaveModelPath
        self.OpenModelPath    = OpenModelPath
        self.numSaveIter      = numSaveIter
        self.kernel_size      = numCh
        self.number_of_layers = number_of_layers
        self.error_states     = self.get_init_Elt_state()
        self.saveModel        = False
        self.openModel        = False 
        self.stateCheck()

    def stateCheck(self):
        if self.SaveModelPath != "":
          print("\n ==> Save model at: " + self.SaveModelPath)
          self.saveModel = True
        if self.OpenModelPath != "":
          print("\n ==> Open model at: " + self.SaveModelPath)
          self.openModel = True

    def hidden_layers_selctor(self,nlayer): 
        h_l_down_in  = self.input_layer_size[nlayer]     
        h_l_top_out  = self.hidden_size     [nlayer]
        h_l_down_out = self.hidden_size     [nlayer]
        h_Elt        = self.Elt_size        [nlayer]
        return h_l_down_in,h_l_top_out,\
               h_l_down_out,h_Elt
    
    def get_init_Elt_state(self):
        errot_list = []
        pooling    =  self.image_size
        for i in xrange(self.number_of_layers-1):
            pooling = pooling/2
        for layer in xrange(self.number_of_layers + 1):
             errot_list.append([1,self.input_layer_size[layer],\
                                pooling*2**(self.number_of_layers-layer),\
                                pooling*2**(self.number_of_layers-layer)])
        return errot_list
    
    def get_init_Elt_tensor(self,state):
        return Variable(torch.randn(state[0],state[1],state[2],state[3])).cuda()

    def save_models(self,model,epoch,typeModel):
        torch.save(model,\
          os.path.join(self.SaveModelPath,\
          typeModel+"_epoch_"+str(epoch)+'.pt'))

    def call_Decoder(self,nlayer,Elt_state,\
                     Rlt_top,Rlt_state,epoch):
            # h = hidden layers 
        R_lt_next    = None 
        Elt_state_in = None
        Decode_lt    = None 
        # return hidden layers size:
        h_l_down_in,  h_l_top_out,\
        h_l_down_out, h_Elt  = self.hidden_layers_selctor(nlayer)
        if Elt_state is None:
          Elt_state_in = self.get_init_Elt_tensor(self.error_states[nlayer+1])
        else:
          Elt_state_in = Elt_state
        if Rlt_top is None:
          Decode_lt = Decoder(h_Elt, 0, h_l_down_out,\
                      self.kernel_size).cuda()
          Elt_state = self.get_init_Elt_tensor(self.error_states[nlayer+1])
          R_lt_next = Decode_lt(Elt_state_in,None,None)
        else:    
          Decode_lt = Decoder(h_Elt, Rlt_top.data.size()[1],\
                      h_l_top_out,self.kernel_size).cuda()
          R_lt_next = Decode_lt(Elt_state_in,\
                      Rlt_top,Rlt_state)
        if self.saveModel == True:
          if epoch%self.numSaveIter == 0:
              self.save_models(Decode_lt,epoch,"Decoder")
        return R_lt_next,Decode_lt.parameters()

    def call_Encoder(self,nlayer,\
                     x_lt,R_lt,first,epoch): 
        E_lt      = None
        Encode_lt = None
        # return hidden layers size:
        h_l_down_in,  h_l_top_out,\
        h_l_down_out, h_Elt  = self.hidden_layers_selctor(nlayer)
        Encode_lt            = Encoder([h_l_down_in,h_l_down_out],\
                                       h_l_down_out,self.kernel_size,\
                                       self.image_size).cuda()
        if first is True: 
           E_lt = Encode_lt(x_lt,R_lt,True)
        else:
           E_lt = Encode_lt(x_lt,R_lt,False) 
        if self.saveModel == True:
          if epoch%self.numSaveIter == 0:
              self.save_models(Encode_lt,epoch,"Encoder")
        return E_lt,Encode_lt.parameters()

    ## Algorithm 1 Calculation of PredNet states (page 4 paper)
    def forward(self,x_t,Elt_state,\
                Rlt_state,epoch): 
      Rlt            = None 
      Rlt_top        = None
      Elt_prev       = None
      parD           = None
      parG           = None
      pGlist         = nn.ParameterList()
      pDlist         = nn.ParameterList()
      Rlt_state_tmp  = [None] * (self.number_of_layers)
      Elt_state_tmp  = [None] * (self.number_of_layers)
      # 1.a) Generative part:
      for layer in reversed(range(0,self.number_of_layers)):
          if layer == self.number_of_layers-1:
            Rlt_state[layer],parD  = self.call_Decoder(layer,Elt_state[layer],\
                                     Rlt_top,Rlt_state[layer],epoch)
            Rlt_top               = Rlt_state[layer]
            Rlt_state_tmp[layer]   = Rlt_state[layer]
          else:
            Rlt_state[layer],parD  = self.call_Decoder(layer,Elt_state[layer],\
                                     Rlt_top,Rlt_state[layer],epoch)
            Rlt_top                = Rlt_state[layer]
            Rlt_state_tmp[layer]   = Rlt_state[layer]
      # 2.a)  Discriminative part:
      for layer in range(0,self.number_of_layers):
          if layer == 0:
            Elt_state[layer],parG  = self.call_Encoder(layer,x_t,\
                                     Rlt_state[layer],True,epoch)
            Elt_state_tmp[layer]   = Elt_state[layer]
          else:         
            Elt_state[layer],parG  = self.call_Encoder(layer,Elt_state[layer-1],\
                                     Rlt_state[layer],False,epoch)     
            Elt_state_tmp[layer]   = Elt_state[layer]
      pDlist.extend(parD)
      pGlist.extend(parG)
      return Elt_state_tmp,Rlt_state_tmp,\
             pDlist,pGlist


## Util functions ##
def getResult(data):
    return torch.chunk(data,2,1)[0][0]
     
def getOptimizer(par,lr):
    optimizer = torch.optim.Adam(par.parameters(),\
                                     lr=lr)
def saveImage(dir_path,ImageTg,itr):
    ImageTg     = np.asarray(ImageTg.data.cpu().numpy())
    ImageTg     = np.transpose(ImageTg,  (2, 1, 0))
    dataPathTg  = os.path.join(dir_path,"Image_"+str(itr)+".jpg")
    cv2.imwrite(dataPathTg,ImageTg)
####

def test():
    #######################
    image_size       = 256
    kernel_size      = 3
    num_layers       = 6
    max_epoch        = 100
    T                = 6
    lr               = 0.1
    SaveTrainImage   = ""
    #####################
    SaveModelPath    = ""
    OpenModelPath    = ""
    numSaveIter      = 0
    ######################
    # This must always be kept at zero !!
    y_lt             = Variable(torch.zeros(1, 6, image_size, image_size)).cuda()
    ######################
    cuda_flag        = True
    totLoss          = 0     
    prednet_model    = PrednetModel(num_layers,image_size,kernel_size,\
                                    SaveModelPath,OpenModelPath,\
                                    numSaveIter)
    print('Create a MSE criterion')
    criterion = nn.MSELoss()
    criterion = criterion.cuda()
    for epoch in range(0,max_epoch):
        # init states:
        Rlt_state  = [None] * (num_layers)
        Elt_state  = [None] * (num_layers)
        listError  = []
        # get seq of images from dataset, now is just a random seq !! 
        x_lt       = Variable(torch.randn(T, 1, 3, image_size, image_size)).cuda()
        # set tmp error value
        E_lt       = Variable(torch.randn(T, 1, 6, image_size, image_size)).cuda()
        outImg     = None 
        pDlist     = 0
        pGlist     = 0
        loss       = 0
        for t in range(0, T):
            print("==> Time step: " + str(t))
            Elt_state,Rlt_state,\
            pDlist,pGlist =  prednet_model(x_lt[t],Elt_state,Rlt_state,epoch)
            # IMPLEMENT: equation number 5 of PredNet paper.
            # Sum of MSE of seq time errors sum_t(error_t-0) !!  
            loss   += criterion(Elt_state[0], y_lt) 
            outImg =  Elt_state[0]   
        loss.backward()
        # clip par on GRU !! 
        torch.nn.utils.clip_grad_norm(pDlist,0.5)
        getOptimizer(pDlist,lr)
        getOptimizer(pGlist,lr)
        totLoss = loss.data[0]/max_epoch
        print("==> Loss at epoch: [" + str(epoch) + "] is: " + str(totLoss))
        
        #print("==> Save last train image:")
        #image = getResult(outImg)
        #saveImage(SaveTrainImage,image,epoch)

if __name__ == '__main__':
   test()








   

