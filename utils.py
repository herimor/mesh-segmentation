import time
import random
from copy import deepcopy
from functools import wraps

import numpy as np
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def time_wrapper(func):
    """
    Wrapper method for calculation of the spent running time
    """
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args[0]._verbose:
            print(f'Run {func.__name__}...')
        start_time = time.time()
        result = func(*args, **kwargs)
        if args[0]._verbose:
            print(f'Shape: {result.shape}\nTime spent: {round(time.time() - start_time, 4)} sec\n')
            
        return result
    return wrapper


def parse_obj(obj_path, seg_path=None):
    # TODO: Add annotation

    with open(obj_path) as f:
        # skip header and get counts
        for _ in range(7): next(f)
        vertex_count = int(f.readline().split(':')[1][1:])
        face_count = int(f.readline().split(':')[1][1:])
        for _ in range(2): next(f)

        vertices = np.zeros((vertex_count, 3))
        for i in range(vertex_count):
            vertices[i] = np.array(list(map(float, f.readline().split()[1:])))

        for _ in range(2): next(f)
            
        polygons = np.zeros((face_count, 3, 3))
        faces = np.zeros((face_count, 3), dtype=int)
        
        for i in range(face_count):
            face = np.array(list(map(int, f.readline().split()[1:]))) - 1
            polygons[i] = vertices[face]
            faces[i] = face

    labels = np.zeros((face_count,), dtype=int)
    if seg_path is not None:
        labels = np.array(list(map(int, open(seg_path).readlines())))

    return polygons, vertices, faces, labels


def init_colors():
    # TODO: Add annotation

    color_set = set()
    while len(color_set) != 100:
        red_ch = random.randint(0, 255)
        blue_ch = random.randint(0, 255)
        green_ch = random.randint(0, 255)

        color_set.add((red_ch, blue_ch, green_ch))
    
    def r2h(x):
        return colors.rgb2hex(tuple(map(lambda y: y / 255., x)))
    
    face_colors = []
    for ch_val in color_set:
        face_colors.append(r2h(ch_val))

    return face_colors


def plot_mesh(polygons,
              data_labels,
              pred_labels,
              transforms=[1, 1, 1],
              scale=(-0.6, 0.6),
              face_colors=None):
    # TODO: Add annotation

    plot_labels = [data_labels, pred_labels]
    
    polygons_transf = deepcopy(polygons)
    polygons_transf[:, :, :] *= transforms
    fig = plt.figure(figsize=(20, 8))
    
    if face_colors is None:
        face_colors = init_colors()
        
    for i, labels in enumerate(plot_labels, start=1):
        cols = [face_colors[lbl] for lbl in labels]
        ax = fig.add_subplot(1, 2, i, projection='3d')
        ax.add_collection3d(Poly3DCollection(verts=polygons_transf,
                                             facecolors=cols,
                                             edgecolor='k',
                                             linewidths=.3))
        #
        # hide axis
        ax.set_xlim3d(scale)
        ax.set_ylim3d(scale)
        ax.set_zlim3d(scale)
        #
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # get rid of the spines
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        # get rid of the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.show()


def calc_mesh(faces, num_edges=3):
    mesh = np.zeros((len(faces), num_edges), dtype=np.uint16)
    connectivity_matrix = np.zeros((len(faces), len(faces)), dtype=np.uint8)

    for i, face in enumerate(faces):
        # iterate by the face edges
        for j in range(-1, 2):
            start_adj_edges = np.where(faces == face[j])[0]
            end_adj_edges = np.where(faces == face[j + 1])[0]
            adj_faces_ids = np.intersect1d(start_adj_edges, end_adj_edges)

            if len(adj_faces_ids[adj_faces_ids != i]) > 0:
                mesh[i][j] = adj_faces_ids[adj_faces_ids != i][0]
                connectivity_matrix[i][mesh[i][j]] = 1

    return mesh, connectivity_matrix


def average_labels(mesh, face_labels):
    _face_labels = []
    for cur_face, adj_faces in enumerate(mesh):
        adj_labels = []
        for adj_face in adj_faces:
            adj_labels.append(face_labels[adj_face])

        _face_labels.append(np.argmax(np.bincount(adj_labels)))

    return _face_labels
