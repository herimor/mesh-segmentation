from pathlib import Path


class Config:
    # save_path = Path('/Users/herimor/Documents/Github/mesh-segmentation')
    save_path = Path('/Users/herimor/Documents/3D_mesh_segmentation')
    proc_category = 'human'
    num_proc = 4
    excluded_set = set(range(261, 281))
    net_config = {
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'margin': 3.,
        'n_epochs': 21,
        'batch_size': 256,
        'num_features': 256,
        'layer_width': 512,
        'do_rate': 0.6,
        'log_interval': 100,
        'scheduler_interval': 7,
        'num_val_samples': 2
    }
    categories_indices = {
        'human': (1, 21),
        'cup': (21, 41),
        'glasses': (41, 61),
        'airplane': (61, 81),
        'ant': (81, 101),
        'chair': (101, 121),
        'octopus': (121, 141),
        'table': (141, 161),
        'teddy': (161, 181),
        'hand': (181, 201),
        'plier': (201, 221),
        'fish': (221, 241),
        'bird': (241, 261),
        'armadillo': (281, 301),
        'bust': (301, 321),
        'mech': (321, 341),
        'bearing': (341, 361),
        'vase': (361, 381),
        'fourleg': (381, 401)
    }
