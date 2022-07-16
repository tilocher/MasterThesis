def reshape(arr, shape):
    permutation = [arr.shape.index(x) for x in shape]
    return arr.permute(permutation)