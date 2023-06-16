from typing import List

from utils import CHARACTERISTICS_TAG, COVID_MASKS_TAG, COVID_PROCESSED_TAG, COVID_TAG, NORMAL_MASKS_TAG, NORMAL_TAG, cov_path, cov_processed_path, cov_masks_path, normal_masks_path, normal_path, normal_processed_path, characteristics_path
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
        super().__init__(COVID_TAG, COVID_TAG, cov_path())


class WBCovidMaskDatasetArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(COVID_TAG, COVID_MASKS_TAG, cov_masks_path())


class WBCovidProcessedDatasetArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(COVID_TAG, COVID_PROCESSED_TAG, cov_processed_path())


class WBNormalDatasetArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(NORMAL_TAG, NORMAL_TAG, normal_path())


class WBNormalMaskDatasetArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(NORMAL_TAG, NORMAL_MASKS_TAG, normal_masks_path())


class WBNormalProcessedDatasetArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(NORMAL_TAG, NORMAL_PROCESSED_TAG, normal_processed_path())


class WBCharacteristicsArtifact(WBDatasetArtifact):
    def __init__(self):
        super().__init__(CHARACTERISTICS_TAG, CHARACTERISTICS_TAG, characteristics_path())
