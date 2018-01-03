# PredNet - ConvGRU

This is an PyTorch implementation of [PredNet](https://arxiv.org/abs/1605.08104) paper.

The thing that differs from the previous implementation is the use of a Conv-GRU 
instead of a Conv-LSTM.

Also, I have optimized the way in which the parameters are updated during training, 
increased training speed with long sequences.

However, the PredNet model is too complex for a realtime application !!

The code is just an example, to extend it at the application level you should create your own training file following the test function created to the inside PredNetModel.py

[PredNet Animation](https://coxlab.github.io/prednet/prednet_animation.html)

## Getting Started

```python

python PredNetModel.py

```

### Prerequisites

A NVIDIA GPU of at least 4 GB of global memory
and linux/windows/mac os.

### Installing

Download Python 3.6 [Anaconda](https://www.anaconda.com/download/#linux)

bash Anaconda-3.x.x-Linux-x86[_64].sh

After accepting the license terms specify the install location (which defaults to ~/anaconda).

Then, you should install PyTorch and OpenCV:

```python

conda install pytorch torchvision -c pytorch

conda install -c conda-forge opencv 

```

## Authors

* **Nicolo Savioli** - *Initial work* 


## MIT License 

Copyright <2018> <Nicolo Savioli>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

