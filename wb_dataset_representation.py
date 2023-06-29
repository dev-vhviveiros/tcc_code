from utils import CHARACTERISTICS_TAG, COVID_MASKS_TAG, COVID_PROCESSED_TAG, COVID_TAG, NORMAL_MASKS_TAG, NORMAL_TAG, abs_path, load_config
from utils import NORMAL_PROCESSED_TAG


class WBDatasetArtifact:
    def __init__(self, parent_tag: str, tag: str, path: str):
        self.parent_tag = parent_tag
        self.tag = tag
        self.path = path
        self.aliases = [parent_tag, tag]

    def wb_artifact_path(self, project_path: str, wdb_alias: str) -> str:
        return '%s/%s:%s' % (project_path, self.tag, wdb_alias)


class WBCovidDatasetArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(COVID_TAG, COVID_TAG, abs_path(load_config("raw_covid_dataset_path")))


class WBCovidMaskDatasetArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(COVID_TAG, COVID_MASKS_TAG, abs_path(load_config("covid_masks_path")))


class WBCovidProcessedDatasetArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(COVID_TAG, COVID_PROCESSED_TAG, abs_path(load_config("covid_processed_path")))


class WBNormalDatasetArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(NORMAL_TAG, NORMAL_TAG, abs_path(load_config("raw_normal_path")))


class WBNormalMaskDatasetArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(NORMAL_TAG, NORMAL_MASKS_TAG, abs_path(load_config("normal_masks_path")))


class WBNormalProcessedDatasetArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(NORMAL_TAG, NORMAL_PROCESSED_TAG, abs_path(load_config("normal_processed_path")))


class WBCharacteristicsArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(CHARACTERISTICS_TAG, CHARACTERISTICS_TAG, abs_path(load_config("characteristics_path")))
