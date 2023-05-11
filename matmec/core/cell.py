from math import gamma
from typing import Union
import numpy as np
from matmec.tool.cell_tool import simi_judge
from copy import deepcopy


class Cell:

    __name__ = 'matmec.core.Cell' 

    def __init__(self, lattvec: np.ndarray =None, scale: float =1.0):
        '''
        The Cell class, can be initialized by giving lattvec and scale.
        '''
        self.propdict = {}
        self.scale = scale
        if lattvec is None:
            lattvec = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])
        else:
            lattvec = np.array(lattvec, dtype=float)
            assert(lattvec.shape == (3, 3)), ValueError("The lattice vector should be (3, 3) matrix")
        self.lattvec = lattvec

    # Cell_property: a
    @property
    def a(self) -> np.ndarray:
        '''
        a of the cell
        '''
        return self.lattvec[0]

    # Cell_property: b
    @property
    def b(self) -> np.ndarray:
        '''
        b of the cell
        '''
        return self.lattvec[1]

    # Cell_property: c
    @property
    def c(self) -> np.ndarray:
        '''
        c of the cell
        '''
        return self.lattvec[2]
    
    # Cell_property: lengths
    @property
    def lengths(self) -> list:
        '''
        Lengths of 3 lattice vectors
        '''
        return [np.linalg.norm(lattvec) for lattvec in self.lattvec]

    # Cell_property: angles
    @property
    def angles(self) -> list:
        '''
        Angles between three lattice vectors, return as alpha, beta and gamma
        '''
        alpha = np.arccos(np.dot(self.lattvec[0], self.lattvec[1])/(self.a*self.b))/(np.pi/180)
        beta = np.arccos(np.dot(self.lattvec[1], self.lattvec[2])/(self.b*self.c))/(np.pi/180)
        gamma = np.arccos(np.dot(self.lattvec[0], self.lattvec[2])/(self.a*self.c))/(np.pi/180)
        return [alpha, beta, gamma]

    # Cell_property: volume
    @property
    def volume(self) -> float:
        """
        Volume of the cell.
        """
        return float(abs(np.dot(np.cross(self.a, self.b), self.c)))

    # Cell_property: scale
    def _get_scale(self):
        return self._get_propdict_value('scale')
    def _set_scale(self, scale: float):
        assert(isinstance(scale, (int, float))), 'Scale should be of type int or float'
        self.set_propdict('scale', scale)
    scale = property(_get_scale, _set_scale, doc='The scale of the lattice vectors')

    # Cell_property: lattvec
    def _get_lattvec(self):
        return self._inplace_get_propdict_value('lattvec')
    def _set_lattvec(self, lattvec: np.ndarray =None):
        assert(isinstance(lattvec, (tuple, list, np.ndarray))), 'lattvec should be of type list or numpy.ndarray'
        lattvec = np.array(lattvec, dtype=float).reshape(3, 3)
        self.set_propdict('lattvec', lattvec)
    lattvec = property(_get_lattvec, _set_lattvec, doc='The lattice vectors of the cell')

    # Cell_property: reciprocal_lattice
    @property
    def reciprocal_lattice(self):
        a = self.a*self.scale
        b = self.b*self.scale
        c = self.c*self.scale
        b1 = np.cross(b, c)/np.dot(a, np.cross(b, c))
        b2 = np.cross(c, a)/np.dot(b, np.cross(c, a))
        b3 = np.cross(a, b)/np.dot(c, np.cross(a, b))
        return Cell([b1, b2, b3])

    # Cell_property: propdict
    def _get_propdict_value(self, name):
        '''
        Return the deepcopy of the property
        '''
        return deepcopy(self.propdict.get(name))
    def _inplace_get_propdict_value(self, name):
        '''
        Return the property, and that can be inplacely changed
        '''
        return self.propdict.get(name)
    def set_propdict(self, name, value):
        self.propdict[name] = value

    # Cell_method: get_len_angle
    def get_len_angle(self):
        '''
        This function calculate the length a,b,c and inter angles of axis a,b,c
        '''
        a = np.linalg.norm(self.lattvec[0])
        b = np.linalg.norm(self.lattvec[1])
        c = np.linalg.norm(self.lattvec[2])
        alpha = np.arccos(np.dot(self.lattvec[0], self.lattvec[1])/(a*b))/(np.pi/180)
        beta = np.arccos(np.dot(self.lattvec[1], self.lattvec[2])/(b*c))/(np.pi/180)
        gamma = np.arccos(np.dot(self.lattvec[0], self.lattvec[2])/(a*c))/(np.pi/180)
        return np.array([a, b, c, alpha, beta, gamma])
    
    # Cell_method: get_cell_shape
    def get_cell_shape(self, len_tolerence: float=1.5e-2, angle_tolerence: float=1):
        '''
        According to the calculated angles and length, give the related Bravis lattice
        cell: Cell type
        len_tolerence: tolerence of length a,b,c unit: 
        angle_tolerence: tolerence of angle alpha,beta,gamma unit: °(degree)
        '''
        [a, b, c, alpha, beta, gamma] = self.get_len_angle()
        # we just distinguish orthorhombic cell and non-orthorhombic cell
        if simi_judge([alpha,beta,gamma, 90], 2, angle_tolerence):
            return 'Orthorhombic'
        else:
            return 'Others'
    
    def is_orthotropic(self):
        return self.get_cell_shape() == 'Orthorhombic'
    
    def get_cartesian_coords(self, 
                             direct_coords: Union[list, np.ndarray]):
        direct_coords = np.array(direct_coords, dtype=float).reshape(-1, 3)
        return np.matmul(direct_coords, self.scale*self.lattvec)
    
    def get_direct_coords(self,
                          cartesian_coords: Union[list, np.ndarray]):
        cartesian_coords = np.array(cartesian_coords, dtype=float).reshape(-1, 3)
        return np.linalg.solve(self.scale*self.lattvec.T, cartesian_coords.T).T

    def __repr__(self):
        '''
        The Cell class.
        '''
        if self.get_cell_shape() == 'Orthorhombic':
            [a, b, c, alpha, beta, gamma] = self.get_len_angle()
            return 'Orthorhombic Cell([%s, %s, %s])' % (a, b, c)
        else:
            return 'Cell(%s)' % self.lattvec.tolist()
    
    def copy(self):
        '''
        Get the copy of this Cell itself.
        '''
        return deepcopy(self)
