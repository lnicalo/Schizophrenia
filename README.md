MLSP 2014 Schizophrenia Classification Challenge
==

[Luis F. Nicolas-Alonso]() (PhD student at University of Valladolid)


Summary
--
The goal of the [competition: http://www.kaggle.com/c/mlsp-2014-mri](http://www.kaggle.com/c/mlsp-2014-mri) was to automatically diagnose subjects with schizophrenia based on multimodal features derived from their magnetic resonance imaging (MRI) brain scans.

This approach is based on Common Spatial Pattern filtering, which is a very often used technique in Brain Computer Interface (BCI) research. 

Common Spatial Pattern (CSP) filtering was originally devised as a spatial filtering technique in BCI reseach. CSP computes mean spatial covariances for each class and solves a generalized eigenvalue problem to find spatial filters. CSP maximizes the difference between classes giving each brain area a different weight. 

Functional Network Connectivity (FNC) are correlation values that summarize the overall connection between independent brain maps over time. In short, FNC features quantify the subject's overall level of 'synchronicity' between brain areas using correlation values. Therefore, spatial covariances can be build from these features.

Dependencies
--
To succesfully run the code the following Python libraries are required: 
* Numpy
* Pandas
* Sklearn
* Scipy

Code description
--
All files are written in Python.

The code firstly builds these spatial covariance matrices on the basis of FNC features. After, it solves a generalized eigenvalue problem to compute spatial filters. All spatial filters are not equally relevant. The revelevance depends on the corresponding eigenvalue. So, they are sorted by eigenvalue. The spatial filters with higher and lower eigenvalue are selected. The number of spatial filters can be configured modifying the variable `m`. 

After spatial filtering, a linear model is used to classify the CSP features. Classification output is provided in a probabilistic way. This is because classification performance is measured using the area under the ROC curve (AUC).

How to generate the solution
--
Just run the file.

Licence
--
This software is distributed under the GNU General Public License (version 3 or later); please refer to the file `LICENSE.txt`, included with the software, for details. 

