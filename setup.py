from json import tool
from setuptools import setup, find_packages

# py_m=['latt', 'atom', 'cell']

# required data file like periodic_table.json
package_data = {'matmec': ['tool/*json']}

setup(
    name="matmec",
    author="Yixuan Hu",
    author_email="yixuanhu97@sjtu.edu.cn",
    packages=find_packages(),
    package_data=package_data,
    install_requires=[ 
        'numpy'
    ]
)