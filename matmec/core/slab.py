from operator import index
from typing import Union
from matmec.core import Latt, Cell
from .atom import Atom, UPPDATE_ITEM
import warnings
from copy import deepcopy
from math import gcd
from sympy import solve, Symbol
import itertools
import numpy as np

atoms_propdict = {'elements':'element', 'poslist':'pos', 'fix':'fix', \
                  'velocity': 'velocity'}
slab_prop = ['cell', 'name', '__direct__', 'propdict', 'natom', 'formula', \
             'vacuum', 'hkl', 'oriented_unit_cell', 'normal', 'surface_area']

class abcvector:
    """
    A helper class to conviently give a direction vector, either directly give or using abc.
    *** Notice abcvector is not same as hkl vector, as the hkl vector should be h*b1 + k*b2 + l*b3
    *** Notice that the normal vector of the hkl plane is not 1/h*a1 + 1/k*a2 + 1/l*a3
    1) can give input as int(will be converted to string), string or list of abc
    2) if cell is provided, then return a*a1 + b*a2 + c*a3, otherwise return [a, b, c]
    Args:
        abc: can either be int or str indicating three int numbers; or can be directly the \
            vectors given by list or np.ndarray, the length should be 3
        cell: only useful when using abc to define the vector. can be of one Cell object or \
            a list of Cell objects
        hklvector: if True, return the hkl vector, otherwise return the normal vector with element-wise mulitplication
        reduce: if True, reduce the vector to the smallest integer
    Return: 
        a np.array vector instance. if hklvector is True, then return the hkl vector, otherwise return the normal vector
    """
    def __new__(self, 
                abc: Union[int, str, list, np.ndarray], 
                cell: Cell = None,
                hklvector: bool = False,
                reduce: bool = False):
        # prepare the abc_list
        if isinstance(abc, int): abc=str(abc)
        if isinstance(abc, str):
            abc_list = self.get_numbers_fromstr(abc)
        elif isinstance(abc, (list, np.ndarray)):
            abc_list = np.array(abc, dtype=int)
            if abc_list.shape == (3, 3):
                return abc_list
            elif abc_list.shape == (3,):
                if reduce:
                    m = gcd(abc_list[0], abc_list[1])
                    m = gcd(m, abc_list[2])
                    abc_list = np.array(abc_list/m, dtype=int)
            else:
                raise ValueError('The input abc should be either (3,) or (3,3) shape')
        else:
            raise ValueError('Input abc should be of int, str, list, np.ndarray type')
        
        if hklvector and cell is None:
            raise ValueError('When using hklvector, cell should be provided')

        # prepare cell
        if cell is None:
            res_vector = abc_list
        else:
            cell = Cell(cell)
            if hklvector:
                # *** 
                reciprocal_lattice = cell.reciprocal_lattice
                # get the norma vector, better to calculate from the reciprocal lattice
                res_vector = reciprocal_lattice.get_cartesian_coords(abc_list)[0]
            else:
                res_vector = abc_list[0]*cell.lattvec[0]+abc_list[1]*cell.lattvec[1]+abc_list[2]*cell.lattvec[2]
        
        return res_vector

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


class Slab(Latt):

    __name__ = 'matmec.core.Slab'

    def __init__(self,
                 hkl: Union[str, list, np.ndarray]=None,
                 vacuum: float=0.0,
                 oriented_unit_cell = None,
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
        '''
        Set the vacuum of the cell, the vacuum will only be output when finally output POSCAR file.
        '''
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
    This part is to overwrite the get and set propdict method defined in Latt, as the properties with same name have minor difference
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
        Translate all the atoms with given origin
        Args:
            origin: the displacement made to the atoms
            asDirect: define whether the given origin is in direct coordinate
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
        '''
        Return a copy of the slab
        '''
        return deepcopy(self)

    @classmethod
    def surface(cls,
                latt, 
                hkl: Union[str, int, list, np.ndarray], 
                layers: int,
                vacuum: float = 10.0, 
                x: Union[str, int, list, np.ndarray] = None, 
                y: Union[str, int, list, np.ndarray] = None,
                max_y_search: int = None,
                eps = 1E-6):
        '''
        ***This method is only surface building method you need to use in this class***
        ***----------------IMPORTANT!!!------------------***
        To transform the cartesian positions into another coordinate, you just need to\
        multiply the arr_pos with the new matrix, for instance using the surface method\
        you get a 3 vectors of the new oriented surface slab to be v1, v2, v3:
        arr_pos = np.matmul(arr_pos, np.array([v1, v2, v3])) # then this transform the coordinate
        as the v1, v2, v3 forms the new coordinate basis, it's a simple coordinate transform
        ***----------------IMPORTANT!!!------------------***
        
        Build a slab with given hkl, layers and vacuum. x, y vectors are optional.
        You can:
        1. supply two x, y vectors and hkl to define your slab
        2. supply hkl and x vector, then the y vector will be automatically searched to \
            make the y vectors as perpendicular to hkl and x as possible, \
            with a maxium search counts of max_y_search.
        3. supply hkl and no x, y vectors. The algorithm will automatically define the x, y vectors.
        Args:
            latt: the Latt instance used to build the slab
            hkl: the hkl of the surface
            layers: the number of layers of the slab
            vacuum: the vacuum thickness of the slab (default is 10.0 Angstrom)
            x: the x vector of the slab (optional)
            y: the y vector of the slab (optional)
            max_y_search: the maximum search counts for the y vector (optional)
            eps: the tolerance for the algorithm (optional)
        Return:
            a Slab instance
        '''
        def if_on_plane(normal, vec, cell, eps):
            '''
            Return if the vec given is perpendicular to normal
            '''
            vec = abcvector(vec)
            vec_vector = cell.get_cartesian_coords(vec)[0]
            on_plane = abs(np.dot(vec_vector, normal)) < eps
            return on_plane
        assert (isinstance(layers, int)), "layers should be of int type"
        hkl = abcvector(hkl)
        cell = latt.cell
        is_orthotropic = cell.is_orthotropic()
        reciprocal_lattice = cell.reciprocal_lattice
        # get the normal vector, better to calculate from the reciprocal lattice
        normal_vector = reciprocal_lattice.get_cartesian_coords(hkl)[0]
        if x is None and y is None:
            # x and y are not specified, then just call general_surface
            return cls.general_surface(latt=latt, 
                                       hkl=hkl, 
                                       layers=layers)
        elif x is not None and y is None:
            # x is given, then y needs to be optimized
            x = abcvector(x, reduce=True)
            # reduce x first, and generate supercell along x direction if necessary
            x_dim = gcd(x[0], gcd(x[1], x[2]))

            # the input x vector should be perpendicular to the normal vector
            x_on_plane = if_on_plane(normal=normal_vector, 
                                     vec=x, 
                                     cell=cell, 
                                     eps=eps)
            if not x_on_plane:
                raise ValueError('Given x is not on the plane defined by hkl, pls check')
            
            if is_orthotropic:
                u1 = Symbol('x')
                u2 = Symbol('y')
                u3 = Symbol('z')
                solution = solve([hkl[0]*u1+hkl[1]*u2+hkl[2]*u3, x[0]*u1+x[1]*u2+x[2]*u3], [u1, u2, u3])
                y = find_int_solution([solution[u1], solution[u2]], u3)
            else:
                x_vec = cell.get_cartesian_coords(x)[0]
                # ideal y direction should be perpendicular to both x and normal
                ideal_y = np.cross(normal_vector, x_vec)/np.linalg.norm(np.cross(normal_vector, x_vec))
                '''
                The algorism is like, 
                1) if max_y_search(int instance) is specified, then will go through all possible combination from -max_y_search
                to max_y_search, and try to find the best vector which is most like perpendicular with ideal_y
                ideal_y is perpendicular to x and normal_vec(plane normal).
                2) if max_y_search(int instance) is not specified, then will simply try [k, -h, 0], [0, l, -k] and [l, 0, -h], 
                and we will use the one with highest consine and lowest length
                '''
                candidates = []
                if max_y_search is None:
                    h, k, l = hkl
                    t1 = np.array([k, -h, 0])
                    t2 = np.array([0, l, -k])
                    t3 = np.array([l, 0, -h])
                    t4 = t1 - t2
                    t5 = t1 - t3
                    t6 = t2 - t3
                    possible_y_candidates = [t1, t2, t3, t4, t5, t6]
                else:
                    max_y_search = 5
                    index_range = sorted(reversed(range(-max_y_search, max_y_search+1)), 
                                        key=lambda x: abs(x))
                    possible_y_candidates = itertools.product(index_range, index_range, index_range)         
                for y in possible_y_candidates:
                    # skip the options for y if y == [0, 0, 0] or linear dependent with hkl or x
                    y_vec = cell.get_cartesian_coords(y)[0]
                    if (not any(y_vec)) or abs(np.linalg.det([x, hkl, y])) < 1E-8:
                        continue
                    y_len = np.linalg.norm(y_vec)
                    # calculate the cosine value between y_vec and ideal y direction
                    cosine = abs(np.dot(y_vec, ideal_y)/y_len)
                    candidates.append([y, cosine, y_len])
                    if cosine % 1 < 1e-6:
                        break
                y, cosine, y_len = max(candidates, key = lambda x: (x[1], -x[2]))
        else:
            # x and y are all given
            x = abcvector(x, reduce=True)
            y = abcvector(y, reduce=True)
            x_on_plane = if_on_plane(normal=normal_vector, 
                            vec=x, 
                            cell=cell, 
                            eps=eps)
            y_on_plane = if_on_plane(normal=normal_vector, 
                            vec=y, 
                            cell=cell, 
                            eps=eps)
            if not x_on_plane:
                raise ValueError('Given x is not on the plane defined by hkl, pls check')
            if not y_on_plane:
                raise ValueError('Given y is not on the plane defined by hkl, pls check')
        # v3 is a vector point from origin to one lattice point on the plane decribed as hx+ky+lz=1
        # the x,y,z can be obtained by extended euclidean algorism 
        p, q = ext_gcd(hkl[1], hkl[2])
        a, b = ext_gcd(p * hkl[1] + q * hkl[2], hkl[0])
        v3 = np.dot([b, a*p, a*q], cell.lattvec)
        print(f'v3 is: {cell.get_direct_coords(v3)[0]}')
        surfaceBasis = np.array([cell.get_cartesian_coords(x)[0], cell.get_cartesian_coords(y)[0], v3], 
                                dtype=float)
        # return a left-handed slab
        if np.linalg.det(surfaceBasis) < 0:
            surfaceBasis = -surfaceBasis
        print(f'y is: {y}')
        oriented_unit_cell, slab = cls.build_surface(latt=latt, 
                                                     surfaceBasis=surfaceBasis, 
                                                     layers=layers)
        
        return cls(hkl = hkl,
            vacuum = vacuum,
            oriented_unit_cell = oriented_unit_cell,
            formula = slab)



    @classmethod
    def general_surface(cls, 
                        latt, 
                        hkl: str, 
                        layers: int, 
                        vacuum: float=10):
        '''
        The method to automatically generate a slab with a given hkl.
        Args:
            latt: the base Latt instance, better to provide in a conventional cell 
                to make the generated slab more like the ones defined in textbook
            hkl: the miller index. Should give as a string. Currently only accept 3 digits.
            layers: how many layers?
            vaccum: the vacuum height
        Return:
            a Slab instance
        '''
        assert (isinstance(layers, int)), "layers should be of int type"
        hkl = abcvector(hkl)
        cell = latt.cell
        
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
            t1 = cell.get_cartesian_coords([k, -h, 0])[0]
            t2 = cell.get_cartesian_coords([0, l, -k])[0]
            t3 = cell.get_cartesian_coords([l, 0, -h])[0]
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
            surfaceBasis = np.array([-v1, -v2, -v3])
        else:
            surfaceBasis = np.array([v1, v2, v3])
        
        oriented_unit_cell, slab = cls.build_surface(latt=latt, 
                                                     surfaceBasis=surfaceBasis, 
                                                     layers=layers)

        return cls(hkl = hkl,
            vacuum = vacuum,
            oriented_unit_cell = oriented_unit_cell,
            formula = slab)

    @staticmethod
    def build_surface(latt, 
                      surfaceBasis, 
                      layers: int):
        '''
        The method used by general_surface and surface to build the slab, you usually don't call this method yourself.
        Args:
            latt: the base Latt instance, better to provide in a conventional cell
            surfaceBasis: the basis of the new surface, a 3*3 matrix
            layers: how many layers?
        Return:
            oriented_unit_cell: the oriented unit cell
            slab: the slab instance
        '''
        oriented_unit_cell = latt.copy().get_supercell([2, 2, 2])

        '''
        The algorithm is like use a new surfaceBasis to Box selection the Slab.
        '''
        newCell = Cell(surfaceBasis)

        # now the oriented_unit_cell is obtained and saved, by changing the current cell into new cell without
        # modifying the coordinates
        oriented_unit_cell.set_cell(newCell, scaleatoms=False)
        oriented_unit_cell.merge_sites()
        oriented_unit_cell.wrap()
        # generate supercell and wrap all the atoms in, and then merge the sites

        slab = oriented_unit_cell.copy()
        
        slab = slab.get_supercell([1, 1, layers])
        # do schimidt orthogonalization to a3
        # in this step, a1 and a2 are not changed, and a3 is only projected onto the direction to the cross \
        # of (a1, a2), so we can directly scaleatoms the cell without changing the atoms cartesian coordinates
        _a1, _a2, _a3 = slab.cell.lattvec

        cross_unit = np.cross(_a1, _a2) / np.linalg.norm(np.cross(_a1, _a2))
        normalized_a3 = np.dot(_a3, cross_unit) * cross_unit

        # _a3 = np.cross(_a1, _a2) * np.dot(_a3, np.cross(_a1, _a2)) / np.linalg.norm(np.cross(_a1, _a2))**2
        slab.set_cell([_a1, _a2, normalized_a3], scaleatoms=False)
        slab.wrap()

        # check if some atoms are overlapped after the redifinition of a3, as some atoms will be wrapped back into the cell
        overlapList = slab.check_overlap(tolerence = 0.1, verbose=True)

        # if overlapped, then according to the periodicity of c axis, move the overlapped atoms along c axis (the unorthogonalized one)
        slab.set_direct(False)

        if overlapList:
            for i in overlapList[0]:
                # determine whether the overlaped atoms are in the top or bottom, if on the top, then move along a3, else move along -a3
                if np.abs(slab.atomlist[i].pos[2] - 1) < 0.5: # because we alreddy wrapped, so the z coordinate is in [0, 1]
                    slab.atomlist[i].move(_a3)
                else:
                    slab.atomlist[i].move(_a3*-1)
        
        slab.set_direct(True)

        # further normalize the cell, to make a1 as parallel with x axis of cartesian coordinate, so as a3 parallel with c axis
        a1 = [np.linalg.norm(_a1), 0, 0]
        # rotate the a2 in xy plane accordingly
        a2 = [np.dot(_a1, _a2) / np.linalg.norm(_a1), np.sqrt(np.linalg.norm(_a2)**2 - (np.dot(_a1, _a2) / np.linalg.norm(_a1))**2), 0]
        a3 = [0, 0, np.linalg.norm(normalized_a3)]
        slab.set_cell([a1, a2, a3], scaleatoms=True)

        # return hkl,vacuum,oriented_unit_cell,slab
        return oriented_unit_cell, slab

    def write_to_poscar(self, 
                        file: str='POSCAR',
                        ifsort: bool=True):
        '''
        Overwrite the write_to_poscar method of Latt class to output the current Slab into a POSCAR file
        '''
        # generate a tmp Latt instance to write the POSCAR file, for the convinience of Vacuum layer
        tmpLatt = self.copy()
        tmpLatt.wrap()
        # generate a tmp cell instance, for the convinience of Vacuum layer
        tmpCell = tmpLatt.cell.copy()
        tmpCell.lattvec[2, 2] += self.vacuum
        # set the tmpLatt cell to tmpCell
        tmpLatt.set_cell(tmpCell, scaleatoms=False)

        s = tmpLatt.to_poscar_string(ifsort=ifsort)
        print('wrote POSCAR into %s' % file)
        with open(file, 'w') as f:
            f.write(s)


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

def find_int_solution(exprs, varies_Symbol):
    '''
    
    '''
    results = []
    num_exprs = len(exprs)
    i = 1
    isInt = False
    while not isInt:
        results_list = np.array([expr.evalf(subs={varies_Symbol:i}, chop=1E-6) for expr in exprs], dtype=float)
        isInt = np.array(results_list % 1 == 0, dtype=bool).all()
        if isInt:
            return np.append(results_list, i)
        i += 1