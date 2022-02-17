# Graph Convolutional Networks for Unstructured Flow Fields

_Ogoke, F., Meidani, K., Hashemi, A., & Farimani, A. B. (2021). Graph convolutional networks applied to unstructured flow field data. Machine Learning: Science and Technology, 2(4), 045020._ 

This repository is the implementation corresponding to the paper "Graph convolutional networks applied to unstructured flow field data", linked [here](https://iopscience.iop.org/article/10.1088/2632-2153/ac1fc9). 


This project applies Graph Convolutional Neural Networks as a framework for dealing with fluid field data on unstructured grids, enabling inference where traditional image-based analysis methods (e.g. traditional Convolutional Neural Networks) are less applicable.


![GCNN Schematic](https://github.com/fogoke/Airfoil-GCNN/blob/3e5abc260924aaf6303b2a2105f6a6378d55e1e1/AirfoilGCNN/figures/figure_2.png)




## Dataset
The data associated with this paper was created through laminar flow field simulations of the aerodynamics of 1200 airfoils, obtained from the UIUC Airfoil Coordinate Database [(link)](https://m-selig.ae.illinois.edu/ads/coord_database.html). These simulations were carried out using the DOLFIN PDE solver, distributed by FEniCS. This software is presented in the following paper:
* A. Logg and G. N. Wells. DOLFIN: Automated Finite Element Computing, ACM Transactions on Mathematical Software 37 (2010) [ACM](https://doi.org/10.1145/1731022.1731030). 


The resulting velocity field and edge connectivity are extracted from the computational mesh for each airfoil. The complete, post-processed dataset of edge connectivity and velocity information can be downloaded directly [here](https://drive.google.com/uc?id=1hjRndZQMaUPTu8IQbDxKeIqc0GES0kYh&export=download) These data files contain the edge connectivty and velocity information for the zero angle of attack (AoA) dataset in the folders ```edges/``` and ```velocity/``` respectively, while the multiple angle of attack (AoA) dataset are listed as ```edges_large/``` and ```velocity_large```. The samples are denoted based on their filename from the UIUC database. Additionally, samples that have been post processed for use with traditional ML methods are also included. The ```[n]manadjvel/``` series contains data where the adjacency matrix is multiplied by the velocity node information, and the closest _n_ nodes to the center of the airfoil (based on Manhattan distance) are used for the prediction task. The ```[n][...]pool``` series contains edge and velocity information for subset of _n_ nodes selected through random sampling.



## Prerequisites 
The following packages are required in order to run the associated code:


* ```torch_geometric==2.0.3```
* ```keras==2.6.0```
* ```tqdm==4.62.3```
* ```torch==1.8.1+cu101```
* ```tensorflow==2.8.0```





These packages can be installed independently, or all at once by running ```pip install -r requirements.txt```. We recommend that these packages are installed in a new conda environment to avoid clashes with existing package installations. Instructions on defining a new conda environment can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), and more information on the Pytorch-Geometric installation process can also be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).



## Usage

The GCNN is contained in AirfoilGCNN/airfoilgcnn.py, and can be run directly with ```python airfoilgcnn.py```. Upon running the python file, the user will be prompted for the dataset of choice for training, as well as the pre-processing options before the training begins. ```airfoilmlp.py```, ```airfoilcnn.py```, and ```airfoilshallowmethods.py``` contain code for the same prediction task, using fully connected neural networks, a convolutional neural network, and a suite of shallow machine learning methods for comparison. The predictions of the GCNN model are saved in the ```results/``` folder for further analysis.





