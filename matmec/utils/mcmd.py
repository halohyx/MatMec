from matmec.core.latt import Latt
import numpy as np
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl

class MCMD:
    def __init__(self, 
                 latt = None,
                 switch_sites = None,
                 structure_file: str = "mcmd_structure.trajectory"):

        if latt is not None:
            self.latt = latt

            parent_index_length = len(latt.atomlist)
            
            if switch_sites is None:
                self.switch_sites = np.array([range(parent_index_length)])
            else:
                self.switch_sites_format_check(switch_sites = switch_sites,
                                            parent_index_length = parent_index_length)
                self.switch_sites = switch_sites
        else:
            self.latt = None
            self.switch_sites = None
        
        self.current_energy = 1E100
        self.energy_list = []
        self.structure_file = structure_file
        self.iter_num = 0
        self.swap_num = 0


    def get_random_switch_index(self, 
               switch_site_index: int):
        '''
        Get two random index from switch sites
        Parameters:
            switch_site_index: int
                the index of switch sites, we could have switch sites like [[1,2], [3, 4]]
                then switch_site_index = 0, 1, determine which switch site we want to roam in
        Returns:
            random_index: list
                two random index from switch sites
        '''
        # 1. get the switch sites
        if switch_site_index >= len(self.switch_sites):
            raise ValueError('switch_site_index should be less than the length of switch_sites')
        
        switch_sites = self.switch_sites[switch_site_index]
        switch_index = np.random.choice(switch_sites, 2, replace = False)

        # 2. check whether the two index are the same element
        count = 0
        while self.latt.elements[switch_index[0]] == self.latt.elements[switch_index[1]]:
            switch_index = np.random.choice(switch_sites, 2, replace = False)
            count += 1
            if count >= 300:
                raise ValueError('The switch sites are all the same element')
        
        return switch_index
    
    def switch(self,
               switch_index):
        '''
        Do the switch between the two index
        Parameters:
            switch_index: list
                two index to be switched
        Returns:
            switched_latt
                The switched latt
        '''
        if len(switch_index) != 2:
            raise ValueError('switch_index should have two index')
        
        tmp_ele_list = deepcopy(self.latt.elements)
        tmp_ele_list[switch_index[0]], tmp_ele_list[switch_index[1]] = tmp_ele_list[switch_index[1]], tmp_ele_list[switch_index[0]]

        switched_latt = deepcopy(self.latt)
        switched_latt.elements = tmp_ele_list

        return switched_latt

    def run(self,
            max_iter = 3000,
            max_swap = 1000):

        # switch_site_index will iter over all possible sites
        possible_sites = len(self.switch_sites)
        switch_site_index = 0

        while self.iter_num <= max_iter and self.swap_num <= max_swap:
            # 1. get the random index
            switch_index = self.get_random_switch_index(switch_site_index = switch_site_index)
            # 2. do the switch
            switched_latt = self.switch(switch_index = switch_index)
            # 3. get the energy of the switched latt
            new_energy = self.get_energy()
            # 4. judge whether to accept
            if self.simulated_annealing_acceptance(energy_current = self.current_energy,
                                                   energy_new = new_energy,
                                                   temperature = 100):
                self.current_energy = new_energy
                self.latt = switched_latt
                self.swap_num += 1
                self.energy_list.append(self.current_energy)
                self.write_structure()
        
            self.iter_num += 1
            switch_site_index = self.iter_num % possible_sites

        if self.iter_num == max_iter:
            print('The max iteration has been reached')
        elif self.swap_num == max_swap:
            print('The max swap has been reached')
        
        self.plot_energy()

        print("The program has been finished") 

    def get_energy(self):
        return np.random.uniform(0, 10)

    def plot_energy(self, 
                    energy_fig_file = 'energy.png'):
        '''
        Plot the energy profile
        Parameters:
            energy_fig_file: str
                the file name of the energy profile figure
        '''
        plt.figure(figsize = (10, 10))
        ax = plt.subplot(111)

        ax.plot(mcmd.energy_list, 
                color = [0.2, 0.2, 0.2], linewidth = 4.0,
                marker = 'o', markersize = 10.0, 
                markeredgewidth = 2.0, markeredgecolor = [0.2, 0.2, 0.2])

        ax.set_xlabel('Iterations', fontdict = {'size': 24, 'family': 'Calibri', 'style': 'italic'})
        ax.set_ylabel('Energy [eV]', fontdict = {'size': 24, 'family': 'Calibri', 'style': 'italic'})

        # set the width of the spines
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['right'].set_linewidth(2.0)

        ax.tick_params(axis = 'both', which = 'major', labelsize = 18)

        plt.show()
        plt.savefig(energy_fig_file, dpi = 300)


    def write_structure(self):
        '''
        Write the structure to the structure file
        '''
        if self.iter_num == 1:
            with open(self.structure_file, 'w') as f:
                f.write(self.latt.to_poscar_string())
        else:
            with open(self.structure_file, 'a') as f:
                f.write(self.latt.to_poscar_string())
    
    @classmethod
    def switch_sites_format_check(cls, 
                                  switch_sites,
                                  parent_index_length):
        '''
        check the switch_sites format
        '''
        # 1. check whether switch_sites is list or numpy.ndarray
        if not isinstance(switch_sites, (list, np.ndarray)):
            raise TypeError('switch_sites must be list of numpy.ndarray')

        # 2. convert to numpy.ndarray if its type is list
        if isinstance(switch_sites, list):
            switch_sites = np.atleast_2d(switch_sites)
        
        # 3. check the shape, index
        if switch_sites.ndim == 2:
            # check whether the index is satisfied, whether the whole index is the subset of atomlist
            tmp_switch_sites = np.unique(switch_sites.flatten())
            is_subset = np.in1d(tmp_switch_sites, np.array(range(parent_index_length))).all()
            if is_subset:
                return True
            else:
                raise ValueError('switch_sites should be the subset of atomlist')
        else:
            raise ValueError('switch_sites should be 2D array')
    
    @classmethod
    def simulated_annealing_acceptance(cls, 
                                        energy_current, 
                                        energy_new, 
                                        temperature):
        
        boltzmann_constant = 8.617e-5

        if energy_new < energy_current:
            return True  # Always accept better solutions
        else:
            delta_energy = energy_new - energy_current
            acceptance_prob = np.exp(-delta_energy / (boltzmann_constant * temperature))
            return np.random.uniform(0, 1) < acceptance_prob

