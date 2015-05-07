Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/


This folder contains application examples for different image analysis tasks.
These algorithms generally download image data from Cytomine-Core, process them
(e.g. segment or classify), and upload results (e.g. annotation geometries
and associated terms (e.g. cell category) or properties (e.g. cell counts))
to Cytomine-core. Annotation objects that are created by these algorithms can then
be proofread using Review modules on Cytomine-WebUI.


---------------------------------------------------------------------------------
1) classification_validation
* Summary: It downloads from Cytomine-Core annotation images from project(s) and evaluates annotation classification models by cross-validation. No model is saved. It generates confusion matrices that are visible in the Cytomine Web UI

* Typical application: object (e.g. cell) classification algorithm evaluation.

* Based on:
Towards generic image classification: an extensive empirical study
Marée et al., TR 2014 http://orbi.ulg.ac.be/handle/2268/175525

* Used in (Note: Another previous implementation/variants were used):
Towards generic image classification: an extensive empirical study
Marée et al., TR 2014 http://orbi.ulg.ac.be/handle/2268/175525

---------------------------------------------------------------------------------
2) classification_model_builder:

* Summary: It downloads from Cytomine-Core annotation images from project(s) and build a annotation classification model which is saved locally.

* Typical application: object (e.g. cell) classification in cytology slides.

* Based on:
- Towards generic image classification: an extensive empirical study
Marée et al., TR 2014 http://orbi.ulg.ac.be/handle/2268/175525

* Used in (Note: Another previous implementation/variants were used):
- Phenotype Classification of Zebrafish Embryos by Supervised Learning
http://orbi.ulg.ac.be/handle/2268/178357
- Evaluation of CellSolutions BestPrep(R) Automated Thin-Layer Liquid-Based Cytology Papanicolaou Slide Preparation and BestCyte(R) Cell Sorter Imaging System
http://orbi.ulg.ac.be/handle/2268/175580


---------------------------------------------------------------------------------
3) classification_prediction:

* Summary: It downloads from Cytomine-Core annotations images from an image (e.g. detected by an object finder), 
apply a classification model (previously saved locally), and  uploads to Cytomine-Core annotation 
terms (in a userjob layer).

* Typical application: cell classification in cytology slides.

* Based on:
- Towards generic image classification: an extensive empirical study
Marée et al., TR 2014 http://orbi.ulg.ac.be/handle/2268/175525

* Used in (Note: Another previous implementation/variants were used):
- Phenotype Classification of Zebrafish Embryos by Supervised Learning
http://orbi.ulg.ac.be/handle/2268/178357
- Evaluation of CellSolutions BestPrep(R) Automated Thin-Layer Liquid-Based Cytology Papanicolaou Slide Preparation and BestCyte(R) Cell Sorter Imaging System
http://orbi.ulg.ac.be/handle/2268/175580


---------------------------------------------------------------------------------
4) segmentation_model_builder:

* Summary: It downloads from Cytomine-Core annotation images+alphamasks from project(s), build a segmentation (pixel classifier) model which is saved locally.

* Typical application: tumor detection in tissues in histology slides.

* Based on:
- Fast Multi-Class Image Annotation with Random Subwindows and Multiple Output Randomized Trees
http://orbi.ulg.ac.be/handle/2268/12205

* Used in:
- A hybrid human-computer approach for large-scale image-based measurements using web services and machine learning
http://orbi.ulg.ac.be/handle/2268/162084?locale=en

---------------------------------------------------------------------------------
5) segmentation_prediction:

* Summary: It applies a segmentation model (previously saved locally) on a whole image and upload geometries to Cytomin-Core.

* Typical application: tumor detection in tissues in histology slides.

* Based on:
- Fast Multi-Class Image Annotation with Random Subwindows and Multiple Output Randomized Trees
http://orbi.ulg.ac.be/handle/2268/12205

* Used in:
- A hybrid human-computer approach for large-scale image-based measurements using web services and machine learning
http://orbi.ulg.ac.be/handle/2268/162084?locale=en


---------------------------------------------------------------------------------
6) landmark_model_builder:

* Summary: It downloads from Cytomine-Core landmark positions from project images, build landmark detection models which are saved locally

* Typical application: Morphometric studies (e.g. in zebrafish/drosphila development)

* Based on:
Paper under preparation

* Used in:
- Evaluation and Comparison of Anatomical Landmark Detection Methods for Cephalometric X-Ray Images: A Grand Challenge
http://dx.doi.org/10.1109/TMI.2015.2412951
- Automatic localization of interest points in zebrafish images with tree-based methods 
http://orbi.ulg.ac.be/handle/2268/99199 (preliminary version of the algorithm)

---------------------------------------------------------------------------------
7) landmark_prediction:

* Summary: It applies landmark detection models to an image and upload landmark coordinates to Cytomine-Core.

* Typical application: Morphometric studies (e.g. in zebrafish/drosphila development)

* Based on:
Paper under preparation

* Used in:
- Evaluation and Comparison of Anatomical Landmark Detection Methods for Cephalometric X-Ray Images: A Grand Challenge
http://dx.doi.org/10.1109/TMI.2015.2412951
- Automatic localization of interest points in zebrafish images with tree-based methods 
http://orbi.ulg.ac.be/handle/2268/99199 (preliminary version of the algorithm)

---------------------------------------------------------------------------------

8) object_finder:

* Summary: It applies an object finder (e.g. threshold algorithm + connected components) to
tiles in a wholeimage and uploads detected geometries to Cytomine-Core (in a userjob layer)

* Typical application: Object detection (e.g. cells) before object classification

* Based on:
Existing algorithms (Otsu, ...)


---------------------------------------------------------------------------------

9) detect_sample:

* Summary: It applies a thresholding algorithm to a thumbnail of a whole image (downloaded
from Cytomine-Core) and upload detected geometries to Cytomine-Core (in a userjob layer)

* Typical application: Detect sample region before applying other algorithms (e.g. segmentation)

* Based on:
Existing algorithms (Adaptive Thresholding)


---------------------------------------------------------------------------------


Future work:
- Object-oriented implementation of applications into a common analysis workflow.
