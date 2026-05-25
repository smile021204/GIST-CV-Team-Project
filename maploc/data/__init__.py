from .gist_abc.dataset import GistABCDataModule
from .mapillary.dataset import MapillaryDataModule

modules = {"mapillary": MapillaryDataModule, "gist_abc": GistABCDataModule}
