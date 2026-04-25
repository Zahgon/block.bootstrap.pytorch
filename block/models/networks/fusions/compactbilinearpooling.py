import types
import torch
import torch.nn as nn
from torch.autograd import Function


def CountSketchFn_forward(h, s, output_size, x, force_cpu_scatter_add=False):
    pass


def CountSketchFn_backward(h, s, x_size, grad_output):
    pass

class CountSketchFn(Function):

    @staticmethod
    def forward(ctx, h, s, output_size, x, force_cpu_scatter_add=False):
        pass


    @staticmethod
    def backward(ctx, grad_output):
        pass

class CountSketch(nn.Module):
    r"""Compute the count sketch over an input signal.

    .. math::

        out_j = \sum_{i : j = h_i} s_i x_i

    Args:
        input_size (int): Number of channels in the input array
        output_size (int): Number of channels in the output sketch
        h (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s (array, optional): Optional array of size input_size of -1 and 1.

    .. note::

        If h and s are None, they will be automatically be generated using LongTensor.random_.

    Shape:
        - Input: (...,input_size)
        - Output: (...,output_size)

    References:
        Yang Gao et al. "Compact Bilinear Pooling" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
        Akira Fukui et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding", arXiv:1606.01847 (2016).
    """

    def __init__(self, input_size, output_size, h = None, s = None):
        raise NotImplementedError

    def forward(self, x):
        pass

def ComplexMultiply_forward(X_re, X_im, Y_re, Y_im):
    pass

def ComplexMultiply_backward(X_re, X_im, Y_re, Y_im, grad_Z_re, grad_Z_im):
    pass

class ComplexMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X_re, X_im, Y_re, Y_im):
        pass

    @staticmethod
    def backward(ctx,grad_Z_re, grad_Z_im):
        pass

class CompactBilinearPoolingFn(Function):

    @staticmethod
    def forward(ctx, h1, s1, h2, s2, output_size, x, y, force_cpu_scatter_add=False):
        pass

    @staticmethod
    def backward(ctx,grad_output):
        pass

class CompactBilinearPooling(nn.Module):
    r"""Compute the compact bilinear pooling between two input array x and y

    .. math::

        out = \Psi (x,h_1,s_1) \ast \Psi (y,h_2,s_2)

    Args:
        input_size1 (int): Number of channels in the first input array
        input_size2 (int): Number of channels in the second input array
        output_size (int): Number of channels in the output array
        h1 (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s1 (array, optional): Optional array of size input_size of -1 and 1.
        h2 (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s2 (array, optional): Optional array of size input_size of -1 and 1.
        force_cpu_scatter_add (boolean, optional): Force the scatter_add operation to run on CPU for testing purposes

    .. note::

        If h1, s1, s2, h2 are None, they will be automatically be generated using LongTensor.random_.

    Shape:
        - Input 1: (...,input_size1)
        - Input 2: (...,input_size2)
        - Output: (...,output_size)

    References:
        Yang Gao et al. "Compact Bilinear Pooling" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
        Akira Fukui et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding", arXiv:1606.01847 (2016).
    """
    def __init__(self, input1_size, input2_size, output_size, h1 = None, s1 = None, h2 = None, s2 = None, force_cpu_scatter_add=False):
        raise NotImplementedError

    def forward(self, x, y = None):
        pass
