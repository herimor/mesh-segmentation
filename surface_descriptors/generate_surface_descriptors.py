from config import Config
from surface_descriptors.mesh import Mesh


if __name__ == '__main__':
    for mesh_num in range(*Config.categories_indices[Config.proc_category]):
        if mesh_num in Config.excluded_set:
            continue

        mesh = Mesh(obj_path=Config.data_path / f'obj/{mesh_num}.obj',
                    save_path=Config.save_path / 'data')
        all_features = mesh.calculate_surface_descriptors(save=True)
        print(f'Shape: {all_features.shape}\nMin: {all_features.min()}\nMax: {all_features.max()}')
