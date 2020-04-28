import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

import torch
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary

from config import Config
from siamese_net.trainer import fit
from siamese_net.losses import OnlineTripletLoss
from siamese_net.datasets import TripletMeshDataset
from siamese_net.networks import TripletNet, EmbeddingNet
from siamese_net.metrics import AverageNonzeroTripletsMetric


class DataContainer:
    def __init__(self, face_descriptors, geo_distances, face_labels):
        self.face_descriptors = face_descriptors
        self.geo_distances = geo_distances
        self.face_labels = face_labels


def load_data(category: str = 'chair', verbose: bool = True):
    """
    Loading of the precomputed surface descriptors, geodesic distances and face labels

    Parameters:
    category: str - name of the loaded category

    Returns:
    face_descriptors: np.ndarray - surface descriptors for each face of the mesh
    geo_distances: np.ndarray - geodesic distances between all faces of the mesh
    face_labels: np.ndarray - labels for all faces of the mesh
    """

    face_labels = []
    geo_distances = []
    face_descriptors = []

    for mesh_num in range(*Config.categories_indices[category]):

        labels_path = Config.data_path / f'seg/init/{mesh_num}.seg'
        assert labels_path.exists(), f'Download face labels for {category} category'

        with open(Config.data_path / f'seg/init/{mesh_num}.seg') as f:
            labels = np.array(list(map(int, f.readlines())))
        if len(face_labels) > 0:
            labels += max(face_labels[-1]) + 1
        face_labels.append(labels)

        geo_dist_path = Config.save_path / f'data/geo_dist/{mesh_num}.npy'
        assert geo_dist_path.exists(), f'Compute geodesic distances for {category} category'
        geo_distances.append(np.load(geo_dist_path))

        face_descriptors_path = Config.save_path / f'data/surf_desc/{mesh_num}.npy'
        assert face_descriptors_path.exists(), f'Compute surface descriptors for {category} category'
        face_descriptors.append(np.load(face_descriptors_path))

    face_labels = np.hstack(face_labels)
    geo_distances = np.vstack(geo_distances)
    face_descriptors = np.vstack(face_descriptors)
    face_descriptors = normalize(face_descriptors)

    if verbose:
        print(f'Shapes (20 models by {geo_distances.shape[1]} faces):\n'
              f'Labels: {face_labels.shape}\n'
              f'Geo. distances: {geo_distances.shape}\n'
              f'Face descriptors: {face_descriptors.shape}\n'
              f'min: {face_descriptors.min()} max: {face_descriptors.max()}\n'
              f'mean: {face_descriptors.mean()} std: {face_descriptors.std()}')

    return DataContainer(face_descriptors, geo_distances, face_labels)


def get_indices(val_meshes, len_descriptors, num_faces, num_val_samples, category):
    min_category_index = Config.categories_indices[category][0]
    val_meshes[1] += num_val_samples - 1
    start, stop = (val_meshes % min_category_index) * num_faces
    val_meshes[1] -= num_val_samples - 1

    val_indices = range(start, stop)
    train_indices = [i for i in range(len_descriptors) if i < start or i >= stop]

    return train_indices, val_indices


def get_data_loaders(data_container, val_meshes, num_val_samples, transform, category, cuda=False):
    """
    Get PyTorch data loaders for train and val sets

    Parameters:
    data_container: DataContainer - provides surface descriptors, geodesic distances and face labels
    cuda: bool - use CUDA if available

    Returns:
    train_loader: DataLoader - train loader
    val_loader: DataLoader - validation loader
    """

    train_indices, val_indices = get_indices(val_meshes=val_meshes,
                                             len_descriptors=len(data_container.face_descriptors),
                                             num_faces=data_container.geo_distances.shape[1],
                                             num_val_samples=num_val_samples,
                                             category=category)

    train_dataset = TripletMeshDataset(face_descriptors=data_container.face_descriptors[train_indices],
                                       geo_distances=data_container.geo_distances[train_indices],
                                       face_labels=data_container.face_labels[train_indices],
                                       transform=transform)

    val_dataset = TripletMeshDataset(face_descriptors=data_container.face_descriptors[val_indices],
                                     geo_distances=data_container.geo_distances[val_indices],
                                     face_labels=data_container.face_labels[val_indices],
                                     transform=transform,
                                     mode='val')

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=Config.net_config['batch_size'], shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=Config.net_config['batch_size'], shuffle=False, **kwargs)

    return train_loader, val_loader, val_indices


def train_embeddings(data_container, val_meshes, cuda, category, num_val_samples, save=True, verbose=True):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x))
    ])

    train_loader, val_loader, val_indices = get_data_loaders(data_container=data_container,
                                                             val_meshes=val_meshes,
                                                             num_val_samples=num_val_samples,
                                                             transform=transform,
                                                             category=category,
                                                             cuda=cuda)

    input_shape = data_container.face_descriptors.shape[1]
    embedding_net = EmbeddingNet(input_shape=input_shape,
                                 output_shape=Config.net_config['num_features'],
                                 layer_width=Config.net_config['layer_width'],
                                 do_rate=Config.net_config['do_rate'])
    model = TripletNet(embedding_net)

    if cuda:
        model.cuda()

    loss_fn = OnlineTripletLoss(Config.net_config['margin'])
    optimizer = optim.Adam(params=model.parameters(),
                           lr=Config.net_config['lr'],
                           weight_decay=Config.net_config['weight_decay'])
    scheduler = lr_scheduler.StepLR(optimizer, Config.net_config['scheduler_interval'])

    if verbose:
        summary(model.embedding_net, (input_shape,))

    losses = fit(train_loader=train_loader,
                 val_loader=val_loader,
                 model=model,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 n_epochs=Config.net_config['n_epochs'],
                 cuda=cuda,
                 log_interval=Config.net_config['log_interval'],
                 metrics=(AverageNonzeroTripletsMetric()),
                 plot_loss_curve=True,
                 plot_interval=Config.net_config['scheduler_interval'])

    model.eval()
    num_faces = data_container.geo_distances.shape[1]
    embeddings = np.empty(shape=(num_faces * num_val_samples, Config.net_config['num_features']), dtype=np.float32)
    for i, face_descriptor in enumerate(tqdm(data_container.face_descriptors[val_indices])):
        embeddings[i] = model.get_embedding(transform(face_descriptor)).detach().numpy()

    if save:
        losses_path = Config.save_path / f'data/loss_curve'
        losses_path.mkdir(parents=True, exist_ok=True)
        np.save(losses_path / f"{category}_{'_'.join(val_meshes)}", losses)

        (Config.save_path / 'trained_models').mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(
            Config.save_path /
            'trained_models' /
            (f"siamese_net_m_{int(Config.net_config['margin'])}_"
             f"in_{input_shape}_"
             f"{category}_{'_'.join(map(str, val_meshes))}")))

        embeddings_path = Config.save_path / 'data/net_embed'
        embeddings_path.mkdir(parents=True, exist_ok=True)

        for i in range(num_val_samples):
            np.save(embeddings_path / str(val_meshes[i]), embeddings[num_faces * i: num_faces * (i + 1)])

    return embeddings


def calculate_category_embeddings(category, save=True, verbose=True):
    all_embeddings = []
    cuda = torch.cuda.is_available()
    data_container = load_data(category, verbose)

    num_val_samples = Config.net_config['num_val_samples']
    category_indices = Config.categories_indices[category]
    indices_range = (category_indices[0], category_indices[1] - (num_val_samples - 1))

    for i in range(*indices_range, num_val_samples):
        val_meshes = np.array([i, i + (num_val_samples - 1)])
        embeddings = train_embeddings(data_container=data_container,
                                      val_meshes=val_meshes,
                                      cuda=cuda,
                                      category=category,
                                      num_val_samples=num_val_samples,
                                      save=save,
                                      verbose=verbose)
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings)


if __name__ == '__main__':
    calculate_category_embeddings(category=Config.proc_category)
