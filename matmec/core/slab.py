from pickletools import float8
from typing import Union
from matmec.core import Latt, Cell
from matmec.utils import hklvector
from .atom import Atom, UPPDATE_ITEM
import warnings
from copy import deepcopy
from math import gcd
import numpy as np

atoms_propdict = {'elements':'element', 'poslist':'pos', 'fix':'fix', \
                  'velocity': 'velocity'}
slab_prop = ['cell', 'name', '__direct__', 'propdict', 'natom', 'formula', \
             'vacuum', 'hkl', 'oriented_unit_cell', 'normal', 'surface_area']

class Slab(Latt):
    def __init__(self,
                 hkl,
                 vacuum,
                 oriented_unit_cell,
                 formula: str=None, 
                 cell: Cell=None, 
                 pos= [0., 0., 0.],
                 fix: bool =[True, True, True],
                 velocity=[0., 0., 0.], 
                 isDirect: bool=True,
                 name: str ='MatMecSlab'):
        super().__init__(formula = formula,
                       cell = cell,
                       pos = pos,
                       fix = fix,
                       velocity = velocity,
                       isDirect = isDirect,
                       name = name)
        self.hkl = hkl
        self.vacuum = vacuum
        self.oriented_unit_cell = oriented_unit_cell


    # Slab_property: vacuum
    def _get_vacuum(self):
        return self._get_propdict_value('vacuum')
    def _set_vacuum(self, vacuum):
        vacuum = np.float16(vacuum)
        self._set_propdict('vacuum', vacuum)
    vacuum = property(_get_vacuum, _set_vacuum, doc='The vacuum thickness of this slab.')

    def add_vacuum(self, vacuum):
        '''
        Add vacuum, if vacuum alreddy exist, then add on it.
        '''
        if not self.vacuum:
            warnings.warn('Current vacuum thickness is %.2f, will add more on this!' % self.vacuum)
        self.vacuum += vacuum

    # Slab_property: oriented_unit_cell
    def _get_oriented_unit_cell(self):
        return self._get_propdict_value('oriented_unit_cell')
    def _set_oriented_unit_cell(self, oriented_unit_cell):
        self._set_propdict('oriented_unit_cell', oriented_unit_cell)
    oriented_unit_cell = property(_get_oriented_unit_cell, _set_oriented_unit_cell, doc='oriented_unit_cell of this slab.')

    # Slab_property: hkl
    def _get_hkl(self):
        return self._get_propdict_value('hkl')
    def _set_hkl(self, hkl):
        self._set_propdict('hkl', hkl)
    hkl = property(_get_hkl, _set_hkl, doc='hkl of this slab.')

    # Slab_property: normal
    @property
    def normal(self):
        normal_vector = np.cross(self.cell.lattvec[0], self.cell.lattvec[1])
        return normal_vector/np.linalg.norm(normal_vector)

    # Slab_property: surface_area
    @property
    def surface_area(self):
        '''
        Return the area of the surface
        '''
        return np.linalg.norm(np.cross(self.cell.lattvec[0], self.cell.lattvec[1]))

    # Slab_property: propdict
    '''
    This part is to overwrite the get and set propdict method defined in Latt, as the \
        properties have minor difference
    '''
    def _get_propdict_value(self, name):
        return deepcopy(self.propdict.get(name))
    def _set_propdict(self, name, value=None):
        if name in atoms_propdict:
            self._update_atom_propdict(name)
        elif name in slab_prop:
            # the properties directly from the latt
            self.propdict[name] = value
        else:
            raise ValueError('Provided name is not one of the latt properties.')
    def _update_atom_propdict(self, name):
        assert(name in atoms_propdict), 'The property doesnt belong to atoms_prop, pls call set_propdict'
        exec('prop_value = [ atom.%s for atom in self.atomlist]' % atoms_propdict[name])
        self.propdict[name]=np.array(locals()['prop_value'])
        if name in UPPDATE_ITEM:
            del UPPDATE_ITEM[name]

    # Slab_method: set_origin
    def set_origin(self, 
                   origin: Union[list, np.ndarray, float, int], 
                   asDirect: bool):
        '''
        Move all the atoms with origin
        '''
        if asDirect:
            self.set_direct(True)
        else:
            self.set_direct(False)
        if isinstance(origin, (float, int)):
            origin = np.array([1, 1, 1])*origin
        elif isinstance(origin, (list, np.ndarray)):
            origin = np.array(origin, dtype=float)
        self.poslist += origin


    # Slab_method: center
    # Slab_method: change_vacuum
    # 

    # define the wrap method again
    def wrap(self, 
             caxis=False):
        '''
        Wrap the atoms into the cell.
        Args:
            caxis: define whether to wrap the atoms position in c axis as well
        '''
        eps = 1E-7
        if len(self.poslist) < self.natom:
            self._update_atom_propdict('poslist')
        if self.__direct__:
            poslist = np.array(self.poslist)
            poslist += eps
            if caxis:
                poslist %= 1.0
            else:
                poslist[:, [0, 1]] %= 1.0
            poslist -= eps
            self.poslist = poslist
        else:
            # if current coordinate is cartesian coordinate, \
            # then calculate the corresponding direct coordinate and apply boundary conditions
            poslist = np.array(self.poslist)
            transformMattrix = np.linalg.inv(self.cell.lattvec*self.cell.scale)
            poslist = np.matmul(np.array(poslist), transformMattrix)
            poslist += eps
            if caxis:
                poslist %= 1.0
            else:
                poslist[:, [0, 1]] %= 1.0
            poslist -= eps
            poslist = np.matmul(poslist, self.cell.lattvec*self.cell.scale)
            self.poslist = poslist

    def copy(self):
        return deepcopy(self)

    def surface(latt, hkl, layers: int, x, y):
        
        pass

    @classmethod
    def general_surface(cls, 
                        latt, 
                        hkl, 
                        layers: int, 
                        vacuum: float=10):
        '''
        The method to automatically generate a slab with a given hkl.
        Args:
            latt: the base Latt instance, better to provide in a conventional cell to \
                make the generated slab more like the ones defined in textbook
            hkl: the miller index. Should give as a string
            layers: how many layers?
            vaccum: the vacuum height
        '''
        cell = latt.cell
        assert (isinstance(layers, int)), "layers should be of int type"
        hkl = hklvector.get_numbers_fromstr(hkl)
        h, k, l = hkl[0], hkl[1], hkl[2]
        his0, kis0, lis0 = hkl == 0
        if his0 and kis0 or his0 and lis0 or kis0 and lis0:
            # if two of the hkl are zero
            if not his0:
                v1, v2, v3 = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
            elif not kis0:
                v1, v2, v3 = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
            elif not lis0:
                v1, v2, v3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            else:
                raise ValueError('The input hkl should not be all 0!')
        else:
            a1 = cell.lattvec[0]
            a2 = cell.lattvec[1]
            a3 = cell.lattvec[2]
            t1 = k*a1-h*a2
            t2 = l*a2-k*a3
            t3 = l*a1-h*a3
            p, q = ext_gcd(k, l)
            k1 = p*np.dot(t1, t2) + q*np.dot(t3, t2)
            k2 = l*np.dot(t1, t2) - k*np.dot(t3, t2)
            if np.abs(k2) > 0:
                c = -np.round(k1/k2)
                p = p + c*l
                q = q - c*k
            v1 = p*t1 + q*t3
            v2 = t2 / abs(gcd(l, k))
            a, b = ext_gcd(p * k + q * l, h)
            # v3 is a vector point from origin to one lattice point on the plane decribed as hx+ky+lz=1
            # the x,y,z can be obtained by extended euclidean algorism 
            v3 = np.dot([b, a*p, a*q], latt.cell.lattvec)
        
        if np.linalg.det([v1, v2, v3]) < 0:
            newCell = np.array([-v1, -v2, -v3])
        else:
            newCell = np.array([v1, v2, v3])
        oriented_unit_cell = latt.copy()
        newCell = Cell(newCell)
        oriented_unit_cell.set_cell(newCell)
        return cls(hkl = hkl,
                   vacuum = vacuum,
                   oriented_unit_cell = oriented_unit_cell,
                   formula = oriented_unit_cell)
        slab.wrap()
        slab = slab.get_supercell([1, 1, layers])

        # do schimidt orthogonalization to a3
        # in this step, a1 and a2 are not changed, and a3 is only projected onto the direction to the cross \
        # of (a1, a2), so we can directly scaleatoms the cell without changing the atoms cartesian coordinates
        _a1, _a2, _a3 = slab.cell.lattvec
        _a3 = np.cross(_a1, _a2) * np.dot(_a3, np.cross(_a1, _a2)) / np.linalg.norm(np.cross(_a1, _a2))**2
        slab.set_direct(False)
        slab.set_cell([_a1, _a2, _a3], scaleatoms=False)

        # further normalize the cell, to make a1 as parallel with x axis, a3 parallel with c axis
        a1 = [np.linalg.norm(_a1), 0, 0]
        a2 = [np.dot(_a1, _a2) / np.linalg.norm(_a1), np.sqrt(np.linalg.norm(_a2)**2 - (np.dot(_a1, _a2) / np.linalg.norm(_a1))**2), 0]
        a3 = [0, 0, np.linalg.norm(_a3)]
        slab.set_cell([a1, a2, a3], scaleatoms=True)

        return slab


def ext_gcd(a, b):
    '''
    Extended Euclidean Algorithm
    x1 = y2
    y1 = x2 - (a//b)*y2
    xn, yn would be 1, 0 or 0, 1
    return x1 and y1
    '''
    if b == 0:
        return 1, 0
    elif a % b == 0:
        return 0, 1
    else:
        x, y = ext_gcd(b, a%b)
        return y, x - (a//b)*y