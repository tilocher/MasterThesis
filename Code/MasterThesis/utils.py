import matplotlib.pyplot as plt
import numpy as np
import torch


def reshape(arr, shape):
    permutation = [arr.shape.index(x) for x in shape]
    return arr.permute(permutation)

def Stich(SegmetedData, Overlaps):

    StichedResult = []

    prev_overlap = 0

    weights = [1,1]

    for (LowerSegment, UpperSegment, Overlap) in zip(SegmetedData[:-1], SegmetedData[1:], Overlaps[1:]):

        if Overlap != 0:

            mean = (weights[0]*LowerSegment[-Overlap:] + weights[1]*UpperSegment[:Overlap])/sum(weights)
            LowerSegment[-Overlap:] = mean


        StichedResult.append(LowerSegment[prev_overlap:])

        prev_overlap = Overlap

    StichedResult.append(UpperSegment[Overlap:])

    StichedResult = torch.cat(StichedResult)


    return StichedResult


def GetSubset(Data, SplitIndex):
    LowerIndices = torch.arange(0, SplitIndex, step=1)
    UpperIndices = torch.arange(SplitIndex, len(Data), step=1)

    LowerSet = torch.utils.data.Subset(Data, LowerIndices)
    UpperSet = torch.utils.data.Subset(Data, UpperIndices)
    return LowerSet, UpperSet

def moving_average(data: torch.Tensor, window_size = 7 ):

    T = data.shape[1]

    average = torch.empty(data.shape)

    for t in range(T):

        upper_limit = min(t + int(window_size/2), T)
        lower_limit = max(t - int(window_size/2), 0)


        slices = data[:,lower_limit:upper_limit+1]
        average[:,t] = slices.mean(1)

    return average



if __name__ == '__main__':
    a = torch.randn(1,1000,2,2)
    moving_average(a)


