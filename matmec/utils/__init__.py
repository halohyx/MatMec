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

class hklvector:
    """
    A helper class to conviently give a direction vector, either directly give or using hkl
    Args:
        hkl: can either be int or str indicating three int numbers; or can be directly the \
            vectors given by list or np.ndarray, the length should be 3
        cell: only useful when using hkl to define the vector. can be of one Cell object or \
            a list of Cell objects
        return: a single vector. when using hkl to define the vectors, it returns a list of 
            vectors (use each cell.lattvec with same hkl to define vectors)
    """
    def __new__(self, hkl, cell: Cell=None):
        if isinstance(hkl, int): hkl=str(hkl)
        if isinstance(hkl, str) and cell is not None:
            hkl_list = self.get_numbers_fromstr(hkl)
            if len(hkl_list) != 3:
                raise ValueError('Invalid input!')
            # big problem here is that actually the normal vector of hkl plane is not h*a1+k*a2+l*a3,
            # but h*b1+k*b2+l*b3, so I'm not sure whether this will cause further problem here
            # BUG------------------------------------------------------------------------------------#
            if isinstance(cell, Cell):
                hkl_vector = hkl_list[0]*cell.lattvec[0]+hkl_list[1]*cell.lattvec[1]+hkl_list[2]*cell.lattvec[2]
            elif isinstance(cell, (list, np.ndarray)):
                hkl_vector = np.array([ hkl_list[0]*c.lattvec[0]+hkl_list[1]*c.lattvec[1]+hkl_list[2]*c.lattvec[2] for c in cell ])
        elif isinstance(hkl, str) and cell is None:
            raise ValueError("If you use hkl to define the vector, pls provide the cell")
        elif isinstance(hkl, (list, np.ndarray)):
            if len(hkl) != 3:
                raise ValueError('Invalid input!')
            hkl_vector = np.array(hkl, dtype=float)
        return hkl_vector

    @staticmethod
    def get_numbers_fromstr(number):
        if isinstance(number, int): number = str(number)
        number_list = []
        index = 0
        while index < len(number):
            if 48 <= ord(number[index]) <=57:
                number_list.append(number[index])
                index += 1
            elif number[index] == "-":
                number_list.append(int("-%s" % number[index+1]))
                index += 2
            else:
                raise ValueError('Invalid input!')
        return np.array(number_list, dtype=int)