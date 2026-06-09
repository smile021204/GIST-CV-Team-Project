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


def _load_custom_dji(cfg):
    from .custom_dji.dataset import CustomDjiDataModule

    return CustomDjiDataModule(cfg)


modules = {
    "mapillary": _load_mapillary,
    "kitti": _load_kitti,
    "gist": _load_gist,
    "gist_abc": _load_gist_abc,
    "custom_dji": _load_custom_dji,
}

# Direct class export for scripts that import the DataModule directly
# (e.g. scripts/visualize_sequential.py). Imported defensively so this file
# stays valid even if the submodule is absent on a given branch.
try:
    from .custom_dji.dataset import CustomDjiDataModule  # noqa: F401
except ImportError:
    pass
