from .mapillary.dataset import MapillaryDataModule

modules = {"mapillary": MapillaryDataModule}

# Each project-specific DataModule below is imported defensively so this file
# stays valid whether or not a given branch has added that DataModule yet. This
# also keeps the merge with parallel feature branches (e.g. finetune/GH adding
# `gist_abc`) trivial -- both registrations co-exist instead of conflicting.
try:
    from .custom_dji.dataset import CustomDjiDataModule
    modules["custom_dji"] = CustomDjiDataModule
except ImportError:
    pass

try:
    from .gist_abc.dataset import GistABCDataModule
    modules["gist_abc"] = GistABCDataModule
except ImportError:
    pass
