
from cell import Cell


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