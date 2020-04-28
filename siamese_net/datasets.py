# Project: https://github.com/adambielski/siamese-triplet
# Author: Adam Bielski https://github.com/adambielski
# License: BSD 3-Clause


import numpy as np
from torch.utils.data import Dataset


class TripletMeshDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
           based on geodesic distances between samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self,
                 face_descriptors,
                 geo_distances,
                 face_labels,
                 transform=None,
                 mode='train',
                 test_seed=21):
        assert mode in ('train', 'val')

        self.face_descriptors = face_descriptors
        self.geo_distances = geo_distances
        self.face_labels = face_labels
        self.num_faces = geo_distances.shape[1]
        self.transform = transform
        self.mode = mode

        self.labels_set = set(self.face_labels)
        self.label_to_indices = {
            label: np.where(self.face_labels == label)[0]
            for label in self.labels_set
        }

        if self.mode == 'val':
            self.test_indices = dict()
            random_state = np.random.RandomState(test_seed)

            for anchor_index in range(self.num_faces):
                anchor_label = self.face_labels[anchor_index]
                positive_indices = self.label_to_indices[anchor_label]
                negative_indices = np.delete(np.arange(len(self.face_labels)), positive_indices)

                positive_index = self.get_index(anchor_index=anchor_index,
                                                indices=positive_indices,
                                                random_state=random_state,
                                                mode='positive')
                negative_index = self.get_index(anchor_index=anchor_index,
                                                indices=negative_indices,
                                                random_state=random_state,
                                                mode='negative')
                self.test_indices[anchor_index] = (positive_index, negative_index)

    def __getitem__(self, index):
        descriptors = self.get_triplet_descriptors(index)
        if self.transform is not None:
            anchor_descr = self.transform(descriptors[0])
            positive_descr = self.transform(descriptors[1])
            negative_descr = self.transform(descriptors[2])
        else:
            anchor_descr, positive_descr, negative_descr = descriptors

        return (anchor_descr, positive_descr, negative_descr), []

    def __len__(self):
        return len(self.face_descriptors)

    def get_index(self, anchor_index, indices, random_state, mode='positive'):
        distances = self.geo_distances[anchor_index, indices % self.num_faces]

        # inverse distances for negative samples
        if mode == 'negative':
            distances = distances.max() - distances

        proba = distances / distances.sum()
        index = random_state.choice(indices, p=proba)
        return index

    def get_triplet_descriptors(self, anchor_index):
        if self.mode == 'test':
            positive_index, negative_index = self.test_indices[anchor_index]
            descriptors = self.face_descriptors[[anchor_index, positive_index, negative_index]]
        else:
            anchor_label = self.face_labels[anchor_index]
            positive_indices = self.label_to_indices[anchor_label]
            negative_indices = np.delete(np.arange(len(self.face_labels)), positive_indices)

            positive_index = self.get_index(anchor_index=anchor_index,
                                            indices=positive_indices,
                                            random_state=np.random,
                                            mode='positive')
            negative_index = self.get_index(anchor_index=anchor_index,
                                            indices=negative_indices,
                                            random_state=np.random,
                                            mode='negative')

            descriptors = self.face_descriptors[[anchor_index, positive_index, negative_index]]

        return descriptors
