import functools
from pathlib import PurePath
import numpy as np
from matmec.core import Cell

class iofunction:
    '''
    Decorator, To make a reader or write accept either string or a file descriptor
    '''
    def __init__(self, mode=str):
        self.mode = mode
    def __call__(self, func):
        @functools.wraps(func)
        def iofunc(file, *args, **kwargs):
            fd = None
            ifopen = isinstance(file, (str, PurePath))
            if ifopen:
                fd = open(file, self.mode)
            else:
                fd = file
            obj = func(fd, *args, **kwargs)
            return obj
        return iofunc

def reader(func):
    return iofunction('r')(func)
def writer(func):
    return iofunction('w')(func)

