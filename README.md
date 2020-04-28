# Semi-supervised segmentation of 3D meshes based on pretrained embeddings

## Installation

```
git clone https://github.com/herimor/mesh-segmentation
cd mesh-segmentation
pip install -r requirements.txt
conda install -c conda-forge igl 

# Setup save_path, data_path and proc_category parameters in Config.py file
# Also setup num_proc parameter (used for multiprocessing calculation of surface descriptors)
```

Download simplified version of [Princeton Benchmark for 3D Mesh Segmentation](https://segeval.cs.princeton.edu)
from [Kaggle](https://www.kaggle.com/herimor/princeton-benchmark-for-3d-mesh-segmentation)
and unzip archive in the root directory of the repository. The next tree of directories is expected:

```
mesh-segmentation
|__data
   |__obj
      |__1.obj
      |__2.obj
      ...
   |__seg
      |__init
         |__1.seg
         |__2.seg
      ...
```

## The project consists of 3 parts:

 * Feature extraction
 * Training of descriptors
 * Segmentation
 
### Feature extraction

Calculates various geometric surface descriptors for 3D mesh such as:

 * Geodesic distances
 * Principal and Gaussian curvature
 * PCA
 * Shape diameter function
 * Average geodesic distance
 * Shape context
 * Spin images
 * Face normals

For calculating of features run the next script:

```
python surface_descriptors/generate_surface_descriptors.py
```

The script outputs geodesic distances and all surface descriptors to the next directories:

```
mesh-segmentation
|__data
   |__geo_dist
      |__1.npy
      |__2.npy
      ...
   |__surf_desc
      |__1.npy
      |__2.npy
      ...
```

Implementation based on [IGL](https://libigl.github.io/libigl-python-bindings/tutorials/) and [GDIST](https://pypi.org/project/gdist/) libraries

The list of features inspired by the next paper: [Learning 3D Mesh Segmentation and Labeling](https://people.cs.umass.edu/~kalo/papers/LabelMeshes/)

### Training of descriptors

Train Siamese network with Triplet loss implemented in PyTorch based on pre-calculated surface descriptors.
After calculation of surface descriptors run the next script:

```
python train_pipeline.py
```

The script will output neural nets state_dicts, pre-trained embeddings and train/eval loss curves to the next directories:

```
mesh-segmentation
|__trained_models
   |__siamese_net_m_3_in_267_human_1_2
   |__siamese_net_m_3_in_267_human_3_4
   ...
|__data
   |__net_embed
      |__1.npy
      |__2.npy
      ...
   |__loss_curve
      |__human_1_2.npy
      |__human_3_4.npy
```

Implementation based on the next [repository](https://github.com/adambielski/siamese-triplet)

### Segmentation

Segmentation of 3D mesh presented by the [Agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering) algorithm.
Next metrics were used for evaluation of the clustering quality:
 * [Homogeneity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html)
 * [Completeness](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html)
 * [V-measure](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html)
 * [Adjusted rand score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)

After training of the neural net and extraction of the embeddings run the next script:

```
python test_pipeline.py
```

It will calculates values of all metrics and output predicted labels for all faces of the 3D mesh in the next directory:

```
mesh-segmentation
|__data
   |__seg
      |__pred
         |__1.seg
         |__2.seg
         ...
```

The code was tested on MacOS High Sierra 10.13