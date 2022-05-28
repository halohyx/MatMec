import numpy as np
from matmec.tool.cell_tool import simi_judge


class Cell:

    __name__ = 'matmec.core.Cell' 

    def __init__(self, scale: float =1.0, lattvec: np.ndarray =None):
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

    def get_len_angle(self):
        '''
        This function calculate the length a,b,c and inter angles of axis a,b,c
        '''
        a = np.linalg.norm(self.lattvec[0])
        b = np.linalg.norm(self.lattvec[1])
        c = np.linalg.norm(self.lattvec[2])
        alpha = np.arccos(np.dot(self.lattvec[0], self.lattvec[1])/(a*c))/(np.pi/180)
        beta = np.arccos(np.dot(self.lattvec[1], self.lattvec[2])/(b*c))/(np.pi/180)
        gamma = np.arccos(np.dot(self.lattvec[0], self.lattvec[2])/(a*b))/(np.pi/180)
        return np.array([a, b, c, alpha, beta, gamma])
    
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

    # property: scale
    def _get_scale(self):
        return self.get_propdict_value('scale')
    def _set_scale(self, scale: float):
        assert(isinstance(scale, (int, float))), 'Scale should be of type int or float'
        self.set_propdict('scale', scale)
    scale = property(_get_scale, _set_scale, doc='The scale of the lattice vectors')

    # property: lattvec
    def _get_lattvec(self):
        return self.get_propdict_value('lattvec')
    def _set_lattvec(self, lattvec: np.ndarray =None):
        assert(isinstance(lattvec, (tuple, list, np.ndarray))), 'lattvec should be of type list or numpy.ndarray'
        lattvec = np.array(lattvec, dtype=float).reshape(3, 3)
        self.set_propdict('lattvec', lattvec)
    lattvec = property(_get_lattvec, _set_lattvec, doc='The lattice vectors of the cell')

    # propdict
    def get_propdict_value(self, name):
        return self.propdict.get(name)
    def set_propdict(self, name, value):
        self.propdict[name] = value


    def __repr__(self):
        if self.get_cell_shape() == 'Orthorhombic':
            [a, b, c, alpha, beta, gamma] = self.get_len_angle()
            return 'Orthorhombic Cell([%s, %s, %s])' % (a, b, c)
        else:
            return 'Cell(%s)' % self.lattvec.tolist()
