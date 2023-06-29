from utils import abs_path, load_config
from glob import glob

# TAG VARIABLES:
DATASET_TAG = "dataset"
COVID_TAG = "covid"
COVID_MASKS_TAG = "covid_mask"
COVID_PROCESSED_TAG = "covid_processed"
NORMAL_TAG = "normal"
NORMAL_MASKS_TAG = "normal_mask"
NORMAL_PROCESSED_TAG = "normal_processed"
CHARACTERISTICS_TAG = "characteristics"
MODEL_TAG = "model"


class DatasetRepresentation:
    """
    A representation of a dataset, containing information about its tag, parent tag, and local path.

    Attributes:
        parent_tag (str): The parent tag of the dataset.
        tag (str): The tag of the dataset.
        path (str): The local path to the dataset (unsynced to wandb).
        aliases (list[str]): A list of aliases for the dataset, including the parent tag and tag.

    Methods:
        images(extension="g"):
            Get a list of all files with the specified extension from the dataset directory.
        wb_artifact_path(project_path, wdb_alias):
            Get the W&B artifact path for the dataset.
    """

    def __init__(self, parent_tag: str, tag: str, path: str):
        """
        Initialize a new DatasetRepresentation object.

        Args:
            parent_tag: The parent tag of the dataset.
            tag: The tag of the dataset.
            path: The local path to the dataset (unsynced to wandb).
        """
        self.parent_tag = parent_tag
        self.tag = tag
        self.path = path
        self.aliases = [parent_tag, tag]

    def images(self, extension: str = "g") -> list[str]:
        """
        Get a list of all files with the specified extension from the dataset directory.

        Args:
            extension: The file extension to search for. Defaults to "g".

        Returns:
            A list of all files with the specified extension from the dataset directory.
        """
        return glob(self.path + f"/*{extension}")

    def wb_artifact_path(self, project_path: str, wdb_alias: str) -> str:
        return '%s/%s:%s' % (project_path, self.tag, wdb_alias)


class CovidDataset(DatasetRepresentation):
    def __init__(self):
        super().__init__(COVID_TAG, COVID_TAG, abs_path(load_config("raw_covid_dataset_path")))


class CovidMaskDataset(DatasetRepresentation):
    def __init__(self):
        super().__init__(COVID_TAG, COVID_MASKS_TAG, abs_path(load_config("covid_masks_path")))


class CovidProcessedDataset(DatasetRepresentation):
    def __init__(self):
        super().__init__(COVID_TAG, COVID_PROCESSED_TAG, abs_path(load_config("covid_processed_path")))


class NormalDataset(DatasetRepresentation):
    def __init__(self):
        super().__init__(NORMAL_TAG, NORMAL_TAG, abs_path(load_config("raw_normal_path")))


class NormalMaskDataset(DatasetRepresentation):
    def __init__(self):
        super().__init__(NORMAL_TAG, NORMAL_MASKS_TAG, abs_path(load_config("normal_masks_path")))


class NormalProcessedDataset(DatasetRepresentation):
    def __init__(self):
        super().__init__(NORMAL_TAG, NORMAL_PROCESSED_TAG, abs_path(load_config("normal_processed_path")))


class Characteristics(DatasetRepresentation):
    def __init__(self):
        super().__init__(CHARACTERISTICS_TAG, CHARACTERISTICS_TAG, abs_path(load_config("characteristics_path")))


class Model(DatasetRepresentation):
    def __init__(self):
        super().__init__(MODEL_TAG, MODEL_TAG, load_config("model_path"))
