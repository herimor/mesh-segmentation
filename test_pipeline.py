import numpy as np
from tqdm import tqdm
from config import Config
from utils import parse_obj, calc_mesh, average_labels

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_rand_score


def calculate_category_scores(category, num_meshes=20, num_metrics=4, save=True, verbose=True):
    labels_path = Config.data_path / 'seg/init'
    assert labels_path.exists(), f'Download face labels for {category} category'

    with open(labels_path / f'{Config.categories_indices[category][0]}.seg', 'r') as f:
        num_faces = len(list(map(int, f.readlines())))

    all_scores = np.empty(shape=(num_meshes, num_metrics))
    all_pred_labels = np.empty(shape=(num_meshes, num_faces), dtype=np.uint8)

    if save:
        (Config.save_path / 'data/seg/pred').mkdir(parents=True, exist_ok=True)

    for i, mesh_num in enumerate(tqdm(range(*Config.categories_indices[category]))):
        embeddings = np.load(f'data/net_embed/{mesh_num}.npy')

        *_, faces, labels = parse_obj(obj_path=Config.save_path / f'data/obj/{mesh_num}.obj',
                                      seg_path=Config.data_path / f'seg/init/{mesh_num}.seg')
        mesh, conn_mat = calc_mesh(faces)

        agl_clust = AgglomerativeClustering(n_clusters=len(np.unique(labels)), connectivity=conn_mat)
        pred_labels = agl_clust.fit_predict(embeddings)

        avg_labels = average_labels(mesh, pred_labels)
        all_pred_labels[i] = avg_labels

        scores = list(homogeneity_completeness_v_measure(labels, avg_labels))
        scores.append(adjusted_rand_score(labels, avg_labels))
        all_scores[i] = scores

        if save:
            with open(Config.save_path / f'data/seg/pred/{mesh_num}.seg', 'w') as f:
                f.writelines(map(lambda l: f'{l}\n', avg_labels))

    if verbose:
        avg_scores = np.average(all_scores, axis=1)
        worst_scores = all_scores[np.argmin(avg_scores)]
        best_scores = all_scores[np.argmax(avg_scores)]

        print(f'Average category score: {round(np.average(avg_scores), 4)}\n'
              f'Average metrics scores:\n'
              f'Homogeneity: {round(np.average(all_scores[:, 0]), 4)}\n'
              f'Completeness: {round(np.average(all_scores[:, 1]), 4)}\n'
              f'V-measure: {round(np.average(all_scores[:, 2]), 4)}\n'
              f'Adjusted rand score: {round(np.average(all_scores[:, 3]), 4)}\n\n'
              f'Best avg. scores in category:\n'
              f'Homogeneity: {round(best_scores[0], 4)}\n'
              f'Completeness: {round(best_scores[1], 4)}\n'
              f'V-measure: {round(best_scores[2], 4)}\n'
              f'Adjusted rand score: {round(best_scores[3], 4)}\n\n'
              f'Worst avg. scores in category:\n'
              f'Homogeneity: {round(worst_scores[0], 4)}\n'
              f'Completeness: {round(worst_scores[1], 4)}\n'
              f'V-measure: {round(worst_scores[2], 4)}\n'
              f'Adjusted rand score: {round(worst_scores[3], 4)}\n')

    return all_scores


if __name__ == '__main__':
    calculate_category_scores(category=Config.proc_category)
