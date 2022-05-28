'''
A class which stores the information of atoms. 
This can be easily used to manipulate atoms, such as tensile, shear, SF generations.
Author: Yixuan Hu
Email: yixuanhu97@sjtu.edu.cn

'''




from multiprocessing.sharedctypes import Value
import os
from tkinter.tix import Tree
from tracemalloc import start
from turtle import tilt
import numpy as np


class Atom:
    '''this a class for a single site'''

    def __init__(self, element, position=[0, 0, 0], xfree=True, yfree=True, zfree=True, 
    index=0) -> None:
        self.element = element
        self.position = position
        self.xfree = xfree
        self.yfree = yfree
        self.zfree = zfree
        self.index = index
        pass

    def __repr__(self) -> str:
        s = '%d: ' % self.index
        s += '%s ' % self.element
        for i in self.position:
            s += '%f ' % float(i)
        if not (self.xfree and self.yfree and self.zfree):
            if self.xfree:
                s += '%s ' % 'T'
            else:
                s += '%s ' % 'F'
            if self.yfree:
                s += '%s ' % 'T'
            else:
                s += '%s ' % 'F'
            if self.zfree:
                s += '%s ' % 'T'
            else:
                s += '%s ' % 'F'
        return s

    def toString(self):
        s = '%d: ' % self.index
        s += '%s ' % self.element
        for x in self.position:
            s += '%f ' % x
        return s[:-1]

    def toStringSelectiveDynamics(self):
        s = '%d: ' % self.index
        s += '%s ' % self.element
        for i in self.position:
            s += '%f ' % float(i)
        if self.xfree:
            s += '%s ' % 'T'
        else:
            s += '%s ' % 'F'
        if self.yfree:
            s += '%s ' % 'T'
        else:
            s += '%s ' % 'F'
        if self.zfree:
            s += '%s ' % 'T'
        else:
            s += '%s ' % 'F'
        return s[:-1]
    
    def printSelectDynamics(self):
        if self.xfree:
            s = '%s ' % 'T'
        else:
            s = '%s ' % 'F'
        if self.yfree:
            s += '%s ' % 'T'
        else:
            s += '%s ' % 'F'
        if self.zfree:
            s += '%s ' % 'T'
        else:
            s += '%s ' % 'F'
        return s[:-1]

    def setElement(self, newEle):
        '''Set the current atom to be a new element'''
        if type(newEle) == str:
            self.element = newEle
        else:
            raise ValueError('Only the str type of element is acceptable!')
    
    def move(self, newPos):
        '''move the current site'''
        self.position = newPos

    def setFree(self, xfree=True, yfree=True, zfree=True):
        '''set the mobility of the current site'''
        self.xfree = xfree
        self.yfree = yfree
        self.zfree = zfree

    def copySite(self, otherSite):
        '''deepcopy the othersite'''
        self.index = otherSite.index
        self.position = otherSite.position
        self.xfree = otherSite.xfree
        self.yfree = otherSite.yfree
        self.zfree = otherSite.zfree

    def equal(self, otherSite):
        '''return if the current site equals the othersite'''
        if (self.index == otherSite.index and self.position == otherSite.position):
            return True
        else:
            return False

class poscar():
    '''
    A class to read and store a poscar file
    '''
    def __init__(self) -> None:
        self.latticeVectors = np.zeros((3, 3), dtype=float)
        self.latticeVectors[:, :] = np.NAN
        self.isSeletiveDynamics = False
        self.systemName = 'HYXdefault11111'
        self.scaleFactor = 1.0
        self.direct = True
        self.elements = []
        self.elementCounts = []
        self.atomsList = []
    
    def __repr__(self) -> str:
        s = '%s \n' % self.systemName
        s += '%f \n' % self.scaleFactor
        for i in self.latticeVectors:
            for j in i:
                s += '%f  ' % j
            s += '\n'
        for i in self.elements:
            s += '    %s  ' % i
        s += '\n'

        for i in self.elementCounts:
            s += '    %s  ' % i
        s +='\n'
        
        if self.isSeletiveDynamics:
            s += 'SelectiveDynamics \n'
            if self.direct:
                s += 'Direct \n'
            else:
                s += 'Cartesian \n'
            for i in self.atomsList:
                for j in i.position:
                    s += '    %f  ' % j
                s += '   %s \n' % i.toStringSelectiveDynamics()[-5:]
        else:
            if self.direct:
                s += 'Direct \n'
            else:
                s += 'Cartesian \n'
            for i in self.atomsList:
                for j in i.position:
                    s += '    %f  ' % j
                s += '\n'
        return s

    def setSystemName(self, name):
        '''
        Directly set the name of the current system, which is the first line of the POSCAR
        '''
        if isinstance(name, str):
            self.systemName = name
        else:
            raise ValueError('New System Name must be a string')

    def setSelectiveDynamics(self, selectiveDynamic: bool =False):
        '''
        Directly set the selective dynamics of the current system
        '''
        if type(selectiveDynamic) != bool:
            raise ValueError('The selective dynamics should be of bool type!')
        self.isSeletiveDynamics = selectiveDynamic
    
    def setScaleFactor(self, scaleFactor):
        '''
        Directly set the scale factor of the current system, which is the second line of the POSCAR
        '''
        if isinstance(scaleFactor, (int, float)):
            self.scaleFactor = scaleFactor
        else:
            raise ValueError('ScaleFactor must be a number')  
    
    def setLatitceVectors(self, newLVs):
        '''
        Directly set the latticeVectors of the current system, should be used immediatly after
        the definetion of the class
        '''
        valid = True
        if len(newLVs) != 3:
            valid = False
            raise ValueError('More than 3 lattice vectors have been input')
        for i in range(len(newLVs)):
            if not isinstance(newLVs[i], (list, np.ndarray)):
                valid = False
                raise ValueError('The input lattice vectors must be a list type')
                break
            if len(newLVs[i]) != 3:
                valid = False
                raise ValueError('The length of each lattice vector must be 3')
        if valid:
            self.latticeVectors = np.array(newLVs)

    def setDirecCoord(self, isDirect: bool =True):
        '''Everytime the direct is changed, the coordinates will be modified'''
        transferNeed = self.direct == isDirect
        if type(isDirect) == bool:
            self.direct = isDirect
        else:
            raise ValueError('isDirect should be of bool type!')
        
        if not transferNeed:
            if self.direct:
                directCoor = self.__cartesianToDirect(self.atomsList, 1)
                self.atomsList = directCoor
            else:
                cartesianCoor = self.__directToCartesian(self.atomsList, 1)
                self.atomsList = cartesianCoor
    
    def __setElements(self, newElements):
        '''
        Directly set the elements of the current system, cannot be used outside the class
        '''
        if isinstance(newElements, list):
            for i in newElements:
                if not isinstance(i, str):
                    raise ValueError('New elements list must be a string type')
        else:
            if not isinstance(newElements, str):
                raise ValueError('New elements must be a string type')
        self.elements = newElements
    
    def __setElementsCounts(self, newElementsCounts):
        '''
        Directly set the elements counts of the current system, cannot be used outside the class
        '''
        if isinstance(newElementsCounts, (list, np.ndarray)):
            for i in newElementsCounts:
                if not isinstance(i, int):
                    raise ValueError('New elements Counts list must be a int type')
        else:
            if not isinstance(newElementsCounts, int):
                raise ValueError('New elements Count must be a int type')
        self.elementCounts = newElementsCounts

    def __mergeSort(self, left, right):
        '''Merge sort algorithm'''
        result = []
        while left and right:
            if left[0] <= right[0]:
                result.append(left.pop(0))
            else:
                result.append(right.pop(0))
        while left:
            result.append(left.pop(0))
        while right:
            result.append(right.pop(0))
        return result
    
    def __merge(self, arr):
        '''Merge sort algorithm'''
        import math
        if len(arr) <= 1:
            return arr
        middle = math.floor(len(arr)/2)
        left = arr[:middle]
        right = arr[middle:]
        return self.__mergeSort(self.__merge(left), self.__merge(right))

    def __checkOverlap(self, atom):
        '''
        Check if the current atom overlap with other atoms in the current system
        '''
        if np.isnan(self.latticeVectors).any():
            for i in self.atomsList:
                distance = np.abs(atom.position - i.position)
                dislength = np.sqrt(np.sum([j**2 for j in distance]))
                if dislength < 0.00000001:
                    raise ValueError('atom %s overlaps with atom %s' % (atom, i))
            return 0

        for i in self.atomsList:
            distance = np.matmul(self.latticeVectors, np.abs(atom.position - i.position))
            dislength = np.sqrt(np.sum([j**2 for j in distance]))
            '''dislength is the absolute length the two atoms'''
            if dislength < 0.0001:
                raise ValueError('atom %s overlaps with atom %s' % (atom, i))
            if  0.0001 <= dislength < 0.1:
                raise Warning('The distance between atom %s overlaps with atom %s is only %f' % (atom, i, dislength))

        return 0

    def __directToCartesian(self, atomsList, mode=1):
        '''Transform the direct coordinates of the given atomsList to cartesian coordinate'''
        '''mode 1 for tranform the current atomsList to cartesian coordinate
           mode 2 for tranform the coordinate given to cartesian coordinate'''
        if np.isnan(self.latticeVectors).any():
            raise ValueError('Found np.nan in lattice verctors, pls reset the current lattice vector')
        transfromMattrix = self.latticeVectors
        
        newAtomsList = atomsList
        if mode == 1:
            for i in newAtomsList:
                i.position = np.matmul(np.array([i.position]), transfromMattrix)[0]
        elif mode == 2:
            newAtomsList = np.matmul(np.array([newAtomsList]), transfromMattrix)[0]
        
        return newAtomsList
        
    def __cartesianToDirect(self, atomsList, mode=1):
        '''Transform the cartesian coordinates of the given atomsList to direct coordinate'''
        '''mode 1 for tranform the current atomsList to direct coordinate
           mode 2 for tranform the coordinate given to direct coordinate'''
        if np.isnan(self.latticeVectors).any():
            raise ValueError('Found np.nan in lattice verctors, pls reset the current lattice vector')
        transfromMattrix = np.linalg.inv(self.latticeVectors)
        newAtomsList = atomsList
        if mode == 1:
            for i in newAtomsList:
                i.position = np.matmul(np.array([i.position]), transfromMattrix)[0]
        elif mode == 2:
            newAtomsList = np.matmul(np.array([newAtomsList]), transfromMattrix)[0]
        return newAtomsList

    def __getNum(self, target: str):
        num = ''
        while 48 <= ord(target[-1]) <= 57:
            num += target[-1]
            target = target[:-1]
        
        return int(num[::-1]), target

    def __boundaryCond(self):
        if self.direct:
            for i in self.atomsList:
                for j in range(3):
                    while i.position[j] > 1.001:
                        i.position[j] -= 1
                    while i.position[j] < -0.001:
                        i.position[j] += 1
        else:
            self.setDirecCoord(True)
            for i in self.atomsList:
                for j in range(3):
                    while i.position[j] > 1.001:
                        i.position[j] -= 1
                    while i.position[j] < -0.001:
                        i.position[j] += 1
            self.setDirecCoord(False)           

    def rotate_BasisVector(self, angle, rotate_vector: list = [0, 0, 1]):
        '''
        Rotate the basis vectors with certain angles with a certain rotate_vector, but current it doesn't work well
        '''
        self.setDirecCoord(False)
        angle_r = (angle/180)*np.pi
        x = rotate_vector[0]
        y = rotate_vector[1]
        z = rotate_vector[2]
        # below is the rotate matrix which forces the current vector rotate around the rotate_vector with specified angle
        # reference: https://baike.baidu.com/item/%E6%97%8B%E8%BD%AC%E7%9F%A9%E9%98%B5/3265181
        rotate_matrix = np.array([[np.cos(angle_r)+(1-np.cos(angle_r))*x**2, (1-np.cos(angle_r))*x*y-np.sin(angle_r)*z, (1-np.cos(angle_r))*x*z+np.sin(angle_r)*y], 
                                [(1-np.cos(angle_r))*y*z+np.sin(angle_r)*z, np.cos(angle_r)+(1-np.cos(angle_r))*y**2, (1-np.cos(angle_r))*y*z-np.sin(angle_r)*x], 
                                [(1-np.cos(angle_r))*z*x-np.sin(angle_r)*y, (1-np.cos(angle_r))*z*y+np.sin(angle_r)*x, np.cos(angle_r)+(1-np.cos(angle_r))*z**2]])
        newLatticeVec = []
        for i in self.latticeVectors:
            newLatticeVec.append(np.matmul(rotate_matrix, i))
        
        self.setLatitceVectors(newLVs=newLatticeVec)
        # self.__boundaryCond()

    def sort(self):
        '''
        Sort the atoms by the name of its alphabet of the first leter
        It's using the merging sorting algorithm refined by __merge and __mergesort
        return the sorted name list of self.atomsList and new sorted Atoms list,
        This function operate not inplacely
        '''
        atomsNameList = [i.element for i in self.atomsList]
        
        for i in range(len(atomsNameList)):
            atomsNameList[i] += str(i)
        # print('atomsnamelist: ', atomsNameList)
        # print('atomslist: ', self.atomsList)
        sortedAtomsNameList = self.__merge(atomsNameList)
        # print('sortedNamesList: ', sortedAtomsNameList)
        sortedAtomsList = []
        for i in range(len(sortedAtomsNameList)):
            
            index, name = self.__getNum(sortedAtomsNameList[i]) # use buidin function __getNum to seperate the 'Ce111' like string
            # print(index,name, self.atomsList[index], 'i: ', i)
            sortedAtomsList.append(self.atomsList[index])
            # print(i+1, sortedAtomsList[-1])
            sortedAtomsList[-1].index = i + 1
        
        # print('sortedAtomsList: ', sortedAtomsList)
        sortedAtomsNameDict = dict()
        for i in sortedAtomsNameList:
            num, i = self.__getNum(i)
            if i not in sortedAtomsNameDict.keys():
                sortedAtomsNameDict[i] = 1
            else:
                sortedAtomsNameDict[i] += 1
            del i

        return sortedAtomsNameDict, sortedAtomsList

    def addAtoms(self, elements: str or list, posList: list, elementCounts: int or list =1, 
                xfree=True, yfree=True, zfree=True, direct=True, check_boundary_cond = False):
        '''
        Add some atoms into the atoms list, this operate inplacely
        
        paras:
        <elements>: str or list of str type, specify the elments you are going to add, should be corresponding to elementCounts
        <posList>:  List of the postions of the given elements, should have the same coordinate type as the type indicated by <direct>
        <elementCounts>: specify how many counts for the given elements, will add corresponding positions to the self.atomsList just like the POSCAR do
        <xfree, yfree, zfree>: to specify the mobility (selectiveDynamics) of all the given atoms
        <direct>: Specify the coordinates of the given atoms, True for direct coordinates and False for cartesian coordinates
        
        return The sorted name list of the atomsList
        '''
        posList = np.array(posList)
        if type(elements) == str:
            if type(elementCounts) != int:
                raise ValueError('Detected not the int type element counts, please ensure the correct input!')
            if len(np.array(posList).shape) == 1:
                if elementCounts != 1:
                    raise ValueError('Length of posLists dont equal elementCounts')
                if direct and not self.direct:
                    posList = self.__directToCartesian(posList, 2)
                elif not direct and self.direct:
                    posList = self.__cartesianToDirect(posList, 2)
                else:
                    pass
                atom = Atom(elements, position=posList, xfree=xfree, yfree=yfree, zfree=zfree, index=len(self.atomsList)+1)
                self.__checkOverlap(atom)
                self.atomsList.append(atom)
                # print(self.atomsList[-2:])
            
            else:
                if elementCounts != len(posList):
                    raise ValueError('Length of posLists dont equal elementCounts')
                if direct and not self.direct:
                    posList = self.__directToCartesian(posList, 2)
                elif not direct and self.direct:
                    posList = self.__cartesianToDirect(posList, 2)
                else:
                    pass
                for atom in posList:
                    atom = Atom(elements, atom, xfree, yfree, zfree, index=len(self.atomsList)+1)
                    self.__checkOverlap(atom)
                    self.atomsList.append(atom)
                    del atom
            
        elif type(elements) == list:
            if type(elementCounts) != list:
                raise ValueError('Check the type of elementCounts and elements, elementCounts should be interger!')
            else:
                for i in elementCounts:
                    if type(i) != int:
                        raise ValueError('Check the type of elementCounts and elements, elementCounts should be interger!')
            if len(elementCounts) != len(elements):
                raise ValueError('The length of elements dont equal the length of elementCounts')
            if direct and not self.direct:
                posList = self.__directToCartesian(posList, 2)
            elif not direct and self.direct:
                posList = self.__cartesianToDirect(posList, 2)
            else:
                pass
            for i in range(len(elements)):
                '''For each element, loop elementCounts[0] times, add elementCounts[0] numbers of positions to the atomsList'''
                for j in range(elementCounts.pop(0)):
                    if type(elements[i]) != str:
                        raise ValueError('Type of element should be string!')
                    atom = Atom(elements[i], posList.pop(0), xfree, yfree, zfree, index=len(self.atomsList)+1)
                    self.__checkOverlap(atom)
                    self.atomsList.append(atom)
                    del atom
        else:
            raise ValueError('elements should be of str of list of str type!')

        sortedNamesDict, self.atomsList = self.sort()
        self.elements = list(sortedNamesDict.keys())
        self.elementCounts = list(sortedNamesDict.values())
        if check_boundary_cond:
            self.__boundaryCond()
        else:
            pass

        return sortedNamesDict
    
    def print_hyx():
        print('huyixuan')

    def del_atoms(self, bywhat: str or int):
        '''
        Delete atoms by index or index lists or by element symbol
        '''
        if type(bywhat) == int: # This means delete a single atom by it's index
            self.atomsList.pop(bywhat - 1)
        elif type(bywhat) == list:
            self.atomsList = [ i for i in self.atomsList if i.index not in bywhat]
        elif type(bywhat) == str:
            if bywhat not in self.elements:
                raise ValueError('The element symbol you specified not in current system!')
            self.atomsList = [ i for i in self.atomsList if i.element != bywhat]
        
        sortedNamesDict, self.atomsList = self.sort()
        print(sortedNamesDict)
        self.elements = list(sortedNamesDict.keys())
        self.elementCounts = list(sortedNamesDict.values())

    def readFromPOSCAR(self, file: str ='POSCAR'):
        if os.path.isfile('%s' % file):
            self.__init__()
            with open(file, 'r') as f:
                count = 1
                line = f.readline()
                self.setSystemName(line.strip()) # First line indicating the system name
                while line:
                    line = f.readline()
                    count += 1
                    if count == 2:
                        self.setScaleFactor(float(line.strip()))
                    elif count == 3:
                        lattceVec = np.array(line.strip().split(), dtype=float)
                        if len(lattceVec) != 3:
                            raise ValueError('length of lattice vector not equals 3!')
                        newLatticeVecs = lattceVec
                    elif 4<= count <=5:
                        lattceVec = np.array(line.strip().split(), dtype=float)
                        if len(lattceVec) != 3:
                            raise ValueError('length of lattice vector not equals 3!')
                        newLatticeVecs = np.vstack((newLatticeVecs, lattceVec))
                    elif count == 6:
                        self.setLatitceVectors(newLatticeVecs)
                        self.__setElements(line.strip().split())
                    elif count == 7:
                        tempElementCounts = [int(i) for i in line.strip().split()]
                        self.__setElementsCounts(tempElementCounts)
                    elif count == 8:
                        '''Selective dynamics or Coordinate specification'''
                        if line[0] in ['S', 's']:
                            self.setSelectiveDynamics(True)
                            line = f.readline()
                            count += 1
                            if line[0] in ['D', 'd']:
                                self.setDirecCoord(True)
                            elif line[0] in ['C', 'c']:
                                self.setDirecCoord(False)
                        elif line[0] in ['D', 'd']:
                            self.setSelectiveDynamics(False)
                            self.setDirecCoord(True)
                        elif line[0] in ['C', 'c']:
                            self.setSelectiveDynamics(False)
                            self.setDirecCoord(False)
                        elementList = []
                        for i in range(len(self.elementCounts)):
                            for j in range(self.elementCounts[i]):
                                elementList.append(self.elements[i])
                    elif count >= 9:
                        if elementList:
                            tempPosList = line.strip().split()
                            if not self.isSeletiveDynamics:
                                tempPos = [float(i) for i in tempPosList]
                                # print(elementList[0], tempPos)
                                self.addAtoms(elementList.pop(0), tempPos, direct=self.direct)
                                # print(self)
                                # print(self.atomsList)

                                del tempPos
                                del tempPosList
                            else:
                                if len(tempPosList) != 6:
                                    raise ValueError('This line: %s got some problem, pls check!' % line)
                                tempPos = [float(i) for i in tempPosList[:3]]
                                mobilityList = [False if i in ['F', 'f'] else True for i in tempPosList[3:]]
                                # print(tempPos, self.direct)
                                self.addAtoms(elementList.pop(0), tempPos, xfree=mobilityList[0], 
                                                yfree=mobilityList[1], zfree=mobilityList[2], direct=self.direct)
                                del mobilityList
                                del tempPos
                                del tempPosList
                        else:
                            break
            self.__boundaryCond()
        else:
            raise ValueError('Please ensure the validity of file path.')

    def toPOSCAR(self, file: os.path):
        '''
        To output the current system into a POSCAR file
        '''
        s = '%s \n' % self.systemName
        s += '%f \n' % self.scaleFactor
        for i in self.latticeVectors:
            for j in i:
                s += '%f  ' % j
            s += '\n'
        for i in self.elements:
            s += '    %s  ' % i
        s += '\n'

        for i in self.elementCounts:
            s += '    %s  ' % i
        s +='\n'
        
        if self.isSeletiveDynamics:
            s += 'SelectiveDynamics \n'
            if self.direct:
                s += 'Direct \n'
            else:
                s += 'Cartesian \n'
            for i in self.atomsList:
                for j in i.position:
                    s += '    %f  ' % j
                s += '   %s \n' % i.toStringSelectiveDynamics()[-5:]
        else:
            if self.direct:
                s += 'Direct \n'
            else:
                s += 'Cartesian \n'
            for i in self.atomsList:
                for j in i.position:
                    s += '    %f  ' % j
                s += '\n'
        
        with open(file, 'w+') as f:
            f.write(s)

    def __fixTopBottom(self, element: str or list =None):
        '''
        Can be used to fix bottom and top atoms. 
        You can use the element options to specify which type of elements to fix
        '''
        atoms_list = []
        if not element:
            element = self.elements
        for atom in self.atomsList:
            if atom.element in element:
                atoms_list.append(atom.index - 1)

        z = np.array([])
        for i in atoms_list:
            z = np.append(z, self.atomsList[i].position[2])

        top = z.max()
        bottom = z.min()

        top_list = []
        bottom_list = []


        print('The TOP and Bottom atoms lists are:')
        for i in atoms_list:
            if np.abs(self.atomsList[i].position[2] - top) < 0.01:
                top_list.append(i)
                print(self.atomsList[i])
            elif np.abs(self.atomsList[i].position[2] - bottom) < 0.01:
                bottom_list.append(i)
                print(self.atomsList[i])



        return top_list + bottom_list

    def fix_mobility(self, xfree: bool, yfree: bool, zfree: bool, element=None, topBottom=False):
        # Give the exiting POSCAR, export the mobility fixed POSCAR, 
        # you can specify if the Top atoms are fixed or not

        self.setSelectiveDynamics(True)
        if topBottom:
            topBottom_list = self.__fixTopBottom(element)
            otherAtoms_list = [ val for val in range(len(self.atomsList)) if val not in topBottom_list]
            for i in topBottom_list:
                self.atomsList[i].setFree(xfree, yfree, False)
        else:
            otherAtoms_list = range(len(self.atomsList))
        
        for i in otherAtoms_list:
            self.atomsList[i].setFree(xfree, yfree, zfree)


        

    def movePart(self, DirectionVec: list, MoveDistance: float, lowlimit: float =-1E100, highlimit: float =1E100):
        '''To move the part (lowlimit< c < highlimit) toward the direction to somedistance(MoveDistance)'''
        '''Notice:
            1. the DirectionVec is the real vector, similar as the latticevector,  the atoms will
            move toward that direction
            2. And this step doesn't fix the selective mobility, please use the fix_mobility function to fix the 
            mobility after the this function
            3. The MoveDistance is defined the cartesian distance after the shortest lattice vector is set to 1 by
            divide the three lattice vectors by the shortest lattice vector length.
            Example: move a part higher than 0.32 of the crystal to <a+b> direction with 0.76
                     moveDirec = -(a.latticeVectors[0]/np.linalg.norm(a.latticeVectors[0]) - a.latticeVectors[1]/np.linalg.norm(a.latticeVectors[1]))
                     a.movePart(moveDirec, 0.7621023553, 0.32)
            '''

        xlength = np.linalg.norm(self.latticeVectors[0])
        ylength = np.linalg.norm(self.latticeVectors[1])
        zlength = np.linalg.norm(self.latticeVectors[2])
        minLength = min(xlength, ylength, zlength)

        transform_Matrix = np.linalg.inv(self.latticeVectors/minLength)
        '''This is the transform matrix for coordinate transformation'''
        normalized_directionVec = DirectionVec/np.linalg.norm(DirectionVec)
        final_DirectionVec = MoveDistance*normalized_directionVec
        xyz = np.matmul(final_DirectionVec, transform_Matrix)
        print(xyz)

        self.setDirecCoord(True)
        for i in self.atomsList:
            if lowlimit< i.position[2] < highlimit:
                i.position[0] += xyz[0]
                i.position[1] += xyz[1]
                i.position[2] += xyz[2]
        self.__boundaryCond()

    def uniaxial_tensile(self, Direction: int, start_elongation: float, end_elongation: float, steps: int, relax:bool =True):
        '''
        Note that the tensile direction should be perpendicular to the plane made of other two lattice vectors,
        and relaxation can be applied on all atoms in the other two direction to minimize the system energy
        Direction: the tensile direction, can only be one of the lattice vectors. 0, 1, 2 represent the lattice vectors
        start_elongation: the starting total elongation rate of the tensile test (from 1, thus > 1 means tensile, < 1 means compress)
        end_elongation: the ending total elongation rate of the tensile test (from 1, thus > 1 means tensile, < 1 means compress)
        steps: how many steps to go to reach the required elongation
        relax: whether to relax the atoms after each step
        There are two types of tensile: (1) relax atoms perpendicular to the tensile direction after each step
                                        (2) fix all direction after each step.
        And the strain is equal to (x1-x0)/x0
        '''
        if Direction not in [0, 1, 2]:
            raise ValueError('Direction should be 0, 1, 2, indicating the 3 lattice vectors')
        else:
            if start_elongation <= 0 or end_elongation <= 0:
                raise ValueError('Check the elongation')
            else:
                self.setDirecCoord(True)
                mobility = [True, True, True]
                # mobility[Direction] = False
                if relax:
                    self.fix_mobility(mobility[0], mobility[1], mobility[2])
                else:
                    self.fix_mobility(False, False, False)
                if steps != 0:
                    each_step = self.latticeVectors[Direction]*(end_elongation - start_elongation)/steps
                self.latticeVectors[Direction] = self.latticeVectors[Direction]*start_elongation # set the cell to start position
                self.toPOSCAR('./Tensile_%s_%5.3f.vasp' % (str(0), start_elongation))
                
                if steps != 0:
                    for i in range(steps):
                        self.latticeVectors[Direction] += each_step
                        self.toPOSCAR('./Tensile_%s_%5.3f.vasp' % (str(i+1), (i+1)*(end_elongation - start_elongation)/steps+start_elongation ))

    def moniclinic_shear(self, tilt_axis: int, toward_axis: int, initialStrain: float, endStrain: float, steps: int, relax=True):
        '''
        This function tilt axis 1 to axis 2 with a shear strain.
        tilt_axis: the lattice vector which is tilted
        toward_axis: the lattice vector which is the direction for shearing
        strain: the final strain
        steps: how many steps to reach the strain
        relax: whether to relax the atoms after each step
        There are two types of tensile: (1) relax atoms perpendicular to the tensile direction after each step
                                        (2) fix all direction after each step.
        Example: tilt zone axis c to a with 0.5 strain
                    strain = Proj[a]C/Proj[c0]C
        '''
        if tilt_axis not in [0, 1, 2] or toward_axis not in [0, 1, 2]:
            raise ValueError('tilt_axis and toward_axis should be 0, 1, 2, indicating the 3 lattice vectors')
        else:
            self.setDirecCoord(True)
            if relax:
                self.fix_mobility(True, True, True)
            else:
                self.fix_mobility(False, False, False)
            tilt_Axis_length = np.linalg.norm(self.latticeVectors[tilt_axis])
            toward_Axis_length = np.linalg.norm(self.latticeVectors[toward_axis])
            initialVector = (initialStrain*tilt_Axis_length/toward_Axis_length)*self.latticeVectors[toward_axis]
            self.latticeVectors[tilt_axis] += initialVector
            dispVector = (endStrain-initialStrain)*(tilt_Axis_length/toward_Axis_length)*self.latticeVectors[toward_axis]
            self.toPOSCAR('./Shear_%s_%5.3f.vasp' % (str(0), initialStrain) )
            if steps != 0:
                each_step = dispVector/steps
                for i in range(steps):
                    self.latticeVectors[tilt_axis] += each_step
                    self.toPOSCAR('./Shear_%s_%5.3f.vasp' % (str(i+1), (initialStrain + (i+1)*(endStrain-initialStrain)/steps) ))