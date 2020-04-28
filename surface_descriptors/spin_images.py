import gc
from functools import partial
from multiprocessing import Pool
from pathlib import Path, PosixPath

import igl
import numpy as np
from tqdm import tqdm

from utils import parse_obj
from config import Config

# Object used in multiprocessing
# Made global due to the huge shape (num_faces, num_faces)
angles = None


def _get_spin_images(normals: np.ndarray,
                     centroids: np.ndarray,
                     med_edge_len: float,
                     n_bins: int,
                     angle_support: float,
                     source_index: int) -> np.ndarray:
    """
    Calculating Spin image (https://pdfs.semanticscholar.org/30c3/e410f689516983efcd780b9bea02531c387d.pdf)

    Python implementation inspired by this Matlab repo:
    https://github.com/balwantraikekutte/noseTipClassification/blob/master/generate_spin_image.m

    Parameters:
    n_bins: int - number of bins in the "spin image" descriptor
    angle_support: float - the maximum angle between normals of the faces that are taken into account
    n_dims: int - number of dimensions of the output array

    Returns:
    features: np.ndarray - "spin image" features
    """

    bin_size = med_edge_len
    image_width = n_bins * bin_size
    features = np.zeros((n_bins, n_bins))

    src_normal = normals[source_index]
    src_centroid = centroids[source_index]

    non_source_mask = np.ones(shape=(len(angles)), dtype=bool)
    non_source_mask[source_index] = False
    support_mask = (angles[source_index] < angle_support) * non_source_mask

    suport_centroids = centroids[support_mask]

    alpha = np.sqrt(np.sum((suport_centroids - src_centroid) ** 2, axis=1) -
                    (np.sum(src_normal * (suport_centroids - src_centroid), axis=1)) ** 2)
    nan_mask = np.logical_not(np.isnan(alpha))
    alpha = alpha[nan_mask]

    beta = np.sum(src_normal * (suport_centroids - src_centroid), axis=1)
    beta = beta[nan_mask]

    row_indices = np.floor((image_width / 2 - beta) / bin_size).astype(np.int32)
    col_indices = np.floor(alpha / bin_size).astype(np.int32)

    a = alpha / bin_size - col_indices
    b = (image_width / 2 - beta) / bin_size - row_indices

    row_mask = np.abs(row_indices) < n_bins - 1
    col_mask = np.abs(col_indices) < n_bins - 1
    indices_mask = row_mask * col_mask

    row_indices = row_indices[indices_mask]
    col_indices = col_indices[indices_mask]
    a = a[indices_mask]
    b = b[indices_mask]

    for row_idx, col_idx, _a, _b in zip(row_indices, col_indices, a, b):
        features[row_idx, col_idx] += (1 - _a) * (1 - _b)
        features[row_idx + 1, col_idx] += _a * (1 - _b)
        features[row_idx, col_idx + 1] += (1 - _a) * _b
        features[row_idx + 1, col_idx + 1] += _a * _b

    return features


def calc_spin_images(normals: np.ndarray,
                     centroids: np.ndarray,
                     med_edge_len: float,
                     n_bins: int = 8,
                     angle_support: float = np.pi / 2,
                     n_dims: int = 2,
                     save_path: PosixPath = None,
                     num_proc: int = 4):
    """
    Calculating spin images

    Parameters:
    save: bool - save computed features or not

    Returns:
    spin_images: np.ndarray - spin images descriptor
    """

    print('Calculating Spin images...')

    func = partial(_get_spin_images, *(normals, centroids, med_edge_len, n_bins, angle_support))

    spin_images = []
    with Pool(processes=num_proc) as p:
        with tqdm(total=len(normals)) as pbar:
            for features in p.imap(func, range(len(normals))):
                spin_images.append(features)
                pbar.update()

    if n_dims == 2:
        spin_images = np.reshape(spin_images, (len(normals), n_bins * n_bins))

    if save_path is not None:
        np.save(str(save_path / 'spin_images'), spin_images)

    gc.collect()
    return spin_images / np.max(spin_images)


if __name__ == '__main__':
    mesh_num = 31

    polygons, vertices, faces, _ = parse_obj(obj_path=Config.save_path / f'data/obj/{mesh_num}.obj')
    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)

    # Implementation of methods from Mesh class for testing only
    def calc_normals(polygons):
        ab = polygons[:, 0] - polygons[:, 1]
        ac = polygons[:, 0] - polygons[:, 2]
        cross = np.cross(ab, ac)

        return cross / np.expand_dims(np.linalg.norm(cross, axis=1), axis=1)

    def calc_angles(normals, eps: float = 1e-8):
        return np.arccos(np.dot(normals, normals.T) / (np.linalg.norm(normals, axis=1) + eps))

    normals = calc_normals(polygons)
    angles = calc_angles(normals)
    centroids = np.mean(polygons, axis=1)
    med_edge_len = np.median(igl.edge_lengths(vertices, faces))

    spin_images = calc_spin_images(normals, centroids, med_edge_len)
