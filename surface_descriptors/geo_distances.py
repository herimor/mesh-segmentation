import gc
from functools import partial
from multiprocessing import Pool

import igl
import gdist
import numpy as np
from tqdm import tqdm

from utils import parse_obj
from config import Config


def _get_geo_dist(vertices, faces, lib, source: int):
    """
    Computing exact geodesic distances between source and all other vertices of the mesh

    Parameters:
    source: int - index of the source vertex

    Returns:
    geo_dist: np.ndarray - geodesic distances
    """

    assert lib in ('gdist', 'igl')

    targets = np.arange(source, len(vertices), dtype=np.int32)
    source = np.array([source], dtype=np.int32)

    if lib == 'gdist':
        distances = gdist.compute_gdist(vertices,
                                        faces,
                                        source_indices=source,
                                        target_indices=targets)
    else:
        distances = igl.exact_geodesic(vertices,
                                       faces,
                                       vs=source,
                                       vt=targets)

    return distances


def calc_geodesic_distances(vertices, faces, lib='gdist', num_proc: int = 4) -> np.ndarray:
    """
    Calculating exact geodesic distances between all vertices of the mesh

    Parameters:
    save: bool - save computed features or not

    Returns:
    geo_dist: np.ndarray - geodesic distances between all vertices
    """

    print('Calculating Geodesic distances...')
    geo_dist = np.empty((len(vertices), len(vertices)), dtype=np.float32)
    func = partial(_get_geo_dist, *(vertices, faces, lib))

    with Pool(processes=num_proc) as p:
        with tqdm(total=len(vertices)) as pbar:
            for i, distances in enumerate(p.imap(func, range(len(vertices + 1)))):
                pbar.update()

                geo_dist[i, i:] = distances
                geo_dist[i:, i] = distances

    gc.collect()
    return geo_dist


if __name__ == '__main__':
    mesh_num = 31
    _, vertices, faces, _ = parse_obj(obj_path=Config.save_path / f'data/obj/{mesh_num}.obj')
    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)

    geo_distances = calc_geodesic_distances(vertices, faces)
