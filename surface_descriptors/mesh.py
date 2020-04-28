import igl
import numpy as np
from tqdm import tqdm
from pathlib import PosixPath

from config import Config
from utils import parse_obj, plot_mesh, time_wrapper
from surface_descriptors.geo_distances import calc_geodesic_distances
from surface_descriptors import spin_images


class Mesh:
    """
    Provides storage of triangular mesh and calculation of surface descriptors
    """

    def __init__(self,
                 obj_path: PosixPath,
                 seg_path: PosixPath = None,
                 descr_path: PosixPath = None,
                 save_path: PosixPath = None,
                 verbose: bool = True):
        """
        Parameters:
        obj_path: PosixPath - path to the .obj mesh file
        seg_path: PosixPath - path to the .seg segmentation file
        descr_path: PosixPath - path to the precomputed surface descriptors
        save_path: PosixPath - path for saving of some computed features
        num_proc: int - number of processes used for parallel computation
        verbose: bool - output progress of descriptors computation or not
        """

        self.save_path = save_path
        self._num_proc = Config.num_proc
        self._verbose = verbose
        self._mesh_name = obj_path.name.split('.')[0]

        self.polygons, vertices, faces, self.labels = \
            parse_obj(obj_path, seg_path)
        self.vertices = np.array(vertices, dtype=np.float64)
        self.faces = np.array(faces, dtype=np.int32)

        self.centroids = np.mean(self.polygons, axis=1)
        self.normals = self._calc_normals()
        self.areas = self._calc_areas()
        self.angles = self._calc_angles()

        self.geo_dist = None
        self.centr_geo_dist = None
        self.med_geo_dist = None
        self.med_edge_len = None
        self.curvatures = None
        self.pca = None
        self.sdf = None
        self.avg_geo_dist = None
        self.shape_context = None
        self.spin_images = None
        self.surface_descriptors = None

        if descr_path is not None:
            self.surface_descriptors = np.load(descr_path)

    def calculate_surface_descriptors(self, save: bool = False):
        """
        Calculating surface descriptors for each mesh face

        Parameters:
        save: bool - save features that requires a long computation time or not

        Returns:
        surface_descriptors: np.ndarray - surface descriptors
        """

        self.centr_geo_dist = self.calc_centroid_geodesic_distances(save=save)
        self.med_geo_dist = np.median(self.centr_geo_dist)
        self.med_edge_len = np.median(igl.edge_lengths(self.vertices, self.faces))

        self.curvatures = self.calc_curvatures()
        self.pca = self.calc_pca()
        self.sdf = self.calc_sdf()
        self.avg_geo_dist = self.calc_average_geodesic_distances()
        self.shape_context = self.calc_shape_context()
        spin_images.angles = self.angles
        self.spin_images = spin_images.calc_spin_images(self.normals, self.centroids, self.med_edge_len)

        self.surface_descriptors = np.hstack((
            self.curvatures,
            self.pca,
            self.sdf,
            self.avg_geo_dist,
            self.shape_context,
            self.centroids,
            self.normals,
            self.spin_images))

        if save:
            assert self.save_path is not None, 'Setup save_path for saving of features'
            (self.save_path / 'surf_desc').mkdir(parents=True, exist_ok=True)
            np.save(str(self.save_path / f'surf_desc/{self._mesh_name}'), self.surface_descriptors)


        return self.surface_descriptors

    def plot_mesh(self,
                  pred_labels: np.ndarray = None,
                  transforms: tuple = (1, 1, 1),
                  scale: tuple = (-0.6, 0.6)):
        """
        Plot 3D mesh with colored segments using pyplot

        Parameters:
        pred_labels: np.ndarray - labels of the mesh faces
        transforms: tuple - transformations for each axis {x, y, z}
        scale: tuple - size scales of plots with ground-truth and predicted labels
        """

        if pred_labels is None:
            pred_labels = self.labels

        plot_mesh(self.polygons, self.labels, pred_labels, transforms, scale)

    def _calc_normals(self) -> np.ndarray:
        """
        Calculating surface normals of each mesh face
        
        N = (AB x AC) / |AB x AC|
        
        Parameters:
        polygons: np.ndarray - polygons of triangular mesh with shape (num_faces, 3, 3)
        
        Returns:
        normals: np.ndarray - normals of each mesh face
        """

        ab = self.polygons[:, 0] - self.polygons[:, 1]
        ac = self.polygons[:, 0] - self.polygons[:, 2]
        cross = np.cross(ab, ac)
        normals = cross / np.expand_dims(np.linalg.norm(cross, axis=1), axis=1)

        return normals

    def _calc_areas(self) -> np.ndarray:
        """
        Calculating surface area of each mesh face
        
        S = |AB x AC| / 2
        
        Parameters:
        polygons: np.ndarray - polygons of triangular mesh with shape (num_faces, 3, 3)
        
        Returns:
        areas: np.ndarray - area of each mesh face
        """

        ab = self.polygons[:, 0] - self.polygons[:, 1]
        ac = self.polygons[:, 0] - self.polygons[:, 2]
        cross = np.cross(ab, ac)
        areas = np.linalg.norm(cross, axis=1) / 2

        return areas

    def _calc_angles(self, eps: float = 1e-8) -> np.ndarray:
        """
        Calculating angles between all normals of the faces on the mesh surface
        
        Parameters:
        normals: np.ndarray(num_faces, 3) - normals for each face
        eps: float - epsilon constant
        
        Returns:
        angles: np.ndarray(num_faces, num_faces)
        """

        return np.arccos(np.dot(self.normals, self.normals.T) /
                         (np.linalg.norm(self.normals, axis=1) + eps))

    def calc_centroid_geodesic_distances(self, save: bool = False) -> np.ndarray:
        """
        Calculating approximate geodesic distances between all centroids of the mesh faces

        Parameters:
        save: bool - save computed features or not
        
        Returns:
        centr_geo_dist: np.ndarray - geodesic distances between all centroids of faces
        """

        if self.centr_geo_dist is not None:
            print('Centroid geodesic distances already precomputed')
            return self.centr_geo_dist

        if self.geo_dist is None:
            self.geo_dist = calc_geodesic_distances(self.vertices, self.faces)

        centr_geo_dist = np.mean(np.mean(self.geo_dist[self.faces], axis=1)[..., self.faces], axis=2)
        np.fill_diagonal(centr_geo_dist, 0.)

        if save:
            assert self.save_path is not None, 'Setup save_path for saving of features'
            (self.save_path / 'geo_dist').mkdir(parents=True, exist_ok=True)
            np.save(str(self.save_path / f'geo_dist/{self._mesh_name}'), centr_geo_dist)

        return centr_geo_dist

    @time_wrapper
    def calc_curvatures(self, max_radius: int = 5, min_radius: int = 2) -> np.ndarray:
        """
        Calculating Principal and Gaussian curvature features of each mesh face
        
        Parameters:
        vertices: np.ndarray - (x, y, z) coordinates of the mesh vertices
        faces: np.ndarray - indices of the mesh vertices forming faces
        areas: np.ndarray - surface areas of the mesh faces
        max_radius: int - controls the size of the neighbourhood used
        min_radius: int - minimum value of radius used in calculation of principal curvatures

        Returns:
        curvatures: np.ndarray - curvature statistics
        """

        curvatures = []
        for radius in range(min_radius, min_radius + max_radius):
            pd1, pd2, pv1, pv2 = igl.principal_curvature(v=self.vertices,
                                                         f=self.faces,
                                                         radius=radius,
                                                         use_k_ring=True)
            pv1 = pv1[self.faces]
            pv2 = pv2[self.faces]

            curv_vals = [
                pv1.mean(axis=1),
                np.abs(pv1).mean(axis=1),
                pv2.mean(axis=1),
                np.abs(pv2).mean(axis=1),
                (pv1 * pv2).mean(axis=1),
                np.abs(pv1 * pv2).mean(axis=1),
                ((pv1 + pv2) / 2).mean(axis=1),
                np.abs((pv1 + pv2) / 2).mean(axis=1),
                (pv1 - pv2).mean(axis=1)
            ]

            curv_vals = np.swapaxes(curv_vals, 0, 1)

            curv_dirs = [
                pd1[self.faces].mean(axis=1),
                pd2[self.faces].mean(axis=1)
            ]

            curv_dirs = np.hstack(curv_dirs)
            curvatures.append(np.hstack((curv_vals, curv_dirs)))

        curvatures = np.hstack(curvatures)

        gauss_curv = igl.gaussian_curvature(self.vertices, self.faces)
        gauss_curv = np.expand_dims(gauss_curv[self.faces].mean(axis=1), axis=1)

        return np.hstack((curvatures, gauss_curv)) * self.areas[:, None]

    def _get_pca(self, face_index: int, eps: float = 1e-7) -> np.ndarray:
        """
        We compute the singular values s1,s2,s3 of the covariance of local face centers (weighted by face area),
        for various geodesic radii (5%, 10%, 20%, 30%, 50% relative to the median of all-pairs geodesic distances),
        and add the following features for each patch.
        
        Kalogerakis, E., Hertzmann, A. and Singh, K., 2010. Learning 3D mesh segmentation and labeling.

        Note: geodesic radii were changed to (15%, 20%, 30%, 40%, 50%)

        Parameters:
        face_index: int - index of the face
        eps: float - epsilon constant used for numerical stability
        
        Returns:
        features: np.ndarray - PCA features of the face
        """

        features = []
        for coeff in [0.2, 0.25, 0.3, 0.4, 0.5]:
            radius = self.med_geo_dist * coeff

            dist_mask = self.centr_geo_dist[face_index] <= radius
            local_face_centers = self.centroids[dist_mask]

            cov_mat = np.cov(local_face_centers, aweights=self.areas[dist_mask], rowvar=False)
            s1, s2, s3 = np.linalg.svd(cov_mat, compute_uv=False, hermitian=True)
            # numerical stability
            s3 += eps

            feature = [
                s1,
                s2,
                s3,
                s1 / (s1 + s2 + s3),
                s2 / (s1 + s2 + s3),
                s3 / (s1 + s2 + s3),
                (s1 + s2) / (s1 + s2 + s3),
                (s1 + s3) / (s1 + s2 + s3),
                (s2 + s3) / (s1 + s2 + s3),
                s1 / s2,
                s1 / s3,
                s2 / s3,
                s1 / s2 + s1 / s3,
                s1 / s2 + s2 / s3,
                s1 / s3 + s2 / s3
            ]

            features.append(feature)

        return np.hstack(features)

    @time_wrapper
    def calc_pca(self):
        """
        Calculating PCA features

        Returns:
        pca: np.ndarray - PCA features
        """

        if self._verbose:
            print('Calculating PCA features...')

        pca = []
        with tqdm(total=len(self.faces)) as pbar:
            for features in map(self._get_pca, range(len(self.faces))):
                pca.append(features)
                pbar.update()

        pca = np.array(pca)
        return pca / pca.max()

    @time_wrapper
    def calc_sdf(self, num_samples: int = 30) -> np.ndarray:
        """
        Calculating shape diameter function and logarithmized version of each mesh face
        
        Parameters:
        num_samples: int - number of samples used for sdf calculation
        
        Returns:
        features: np.ndarray - SDF features
        """

        features = []
        sdf = igl.shape_diameter_function(v=self.vertices,
                                          f=self.faces,
                                          p=self.centroids,
                                          n=self.normals,
                                          num_samples=num_samples)
        features.append(sdf)

        norm_sdf = (sdf - sdf.min()) / (sdf.max() - sdf.min())
        for alpha in [1, 2, 4, 8]:
            features.append(np.log(norm_sdf * alpha + 1) / np.log(alpha + 1))

        return np.stack(features, axis=1)

    @time_wrapper
    def calc_average_geodesic_distances(self) -> np.ndarray:
        """
        Calculating average geodesic distance (AGD)
        
        The AGD for each face is computed by averaging the geodesic distance 
        from its face center to all the other face centers.
        In our case, we also consider the squared mean and the 10th, 20th, ..., 90th percentile.
        Then, we normalize each of these 11 statistical measures by subtracting its minimum over all faces.
        
        Kalogerakis, E., Hertzmann, A. and Singh, K., 2010. Learning 3D mesh segmentation and labeling.
        
        Parameters:
        num_samples: int - number of samples used for sdf calculation
        
        Returns:
        features: np.ndarray - AGD features
        """

        features = []
        mean = np.mean(self.centr_geo_dist, axis=1)
        features.append(mean)
        features.append(mean ** 2)

        if self._verbose:
            print('Calculating q-th percentiles...')

        for q in tqdm(range(10, 100, 10)):
            features.append(np.percentile(self.centr_geo_dist, q, axis=1))

        features = np.stack(features, axis=1)
        return features - np.min(features, axis=0)

    @time_wrapper
    def calc_shape_context(self,
                           n_dims: int = 2,
                           n_dist_bins: int = 5,
                           n_angle_bins: int = 6) -> np.ndarray:
        """
        Calculating Shape context (https://en.wikipedia.org/wiki/Shape_context)
        
        For each face, we measure the distribution of all the other faces (weighted by their area)
        in logarithmic geodesic distance bins. And uniform angle bins, where angles are measured
        relative to the normal of each face. We use 5 geodesic distance bins and 6 angle bins.
        
        Kalogerakis, E., Hertzmann, A. and Singh, K., 2010. Learning 3D mesh segmentation and labeling.
        
        Parameters:
        n_dims: int - number of dimensions of the output array
        n_dist_bins: int - number of bins in the histogram of geodesic distances
        n_angle_bins: int - number of bins in the histogram of angles between normals
        
        Returns:
        features: np.ndarray - shape context features
        """

        max_dist = self.centr_geo_dist.max()
        min_dist = self.centr_geo_dist[self.centr_geo_dist > 0.].min()
        dist_bins = np.logspace(np.log10(min_dist), np.log10(max_dist), n_dist_bins + 1)

        max_angle = self.angles.max()
        min_angle = self.angles.min()
        angle_bins = np.linspace(min_angle, max_angle, n_angle_bins + 1)

        features = np.empty((len(self.faces), n_dist_bins, n_angle_bins))

        with tqdm(total=len(self.faces)) as pbar:
            for i, (dist, angle) in enumerate(zip(self.centr_geo_dist, self.angles)):
                for j in range(len(dist_bins) - 1):
                    bin_indices = (dist > dist_bins[j]) * (dist < dist_bins[j + 1])
                    bin_angles = angle[bin_indices]
                    bin_areas = self.areas[bin_indices]

                    hist, _ = np.histogram(bin_angles, bins=angle_bins, weights=bin_areas)
                    features[i, j] = hist

                pbar.update()

        if n_dims == 2:
            features = features.reshape((len(features), n_dist_bins * n_angle_bins))

        return features
