# MatMec
Yixuan's first attempt to build a package.

**MatMec** is a package for easily manipulating atoms for mechanics calculation in computational matirial science. Currently, only VASP form input and output are supported,
you can read and write POSCAR type file by using **read_from_poscar** and **write_to_poscar**.

You can easily generate structures for calculation such as the Generalized Stacking Fault Energy(GSFE) by using the 
function **move_Part** implemented in class Latt, or ideal tensile and shear by **uniaxial_tensile** and **moniclinic_shear**.

Or generating a desired slab structure using the Slab class.

Read the trajectory of the AIMD results reported by VASP and plot it.

Using the SQS in util to generate a random occupied structure. 

This is just a begginning of **MatMec**, a lot of function related to the mechanical properties will be updated soon.

Author: Yixuan Hu  
E-mail: yixuanhu97@sjtu.edu.cn

