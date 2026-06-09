def _load_kitti(cfg):
    from .kitti.dataset import KittiDataModule

    return KittiDataModule(cfg)


def _load_mapillary(cfg):
    from .mapillary.dataset import MapillaryDataModule

    return MapillaryDataModule(cfg)


def _load_gist(cfg):
    from .gist.dataset import GISTDataModule

    return GISTDataModule(cfg)


def _load_gist_abc(cfg):
    from .gist_abc.dataset import GistABCDataModule

    return GistABCDataModule(cfg)


modules = {
    "mapillary": _load_mapillary,
    "kitti": _load_kitti,
    "gist": _load_gist,
    "gist_abc": _load_gist_abc,
}
