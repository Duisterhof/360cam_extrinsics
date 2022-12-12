
from mvs_utils.metadata_reader import MetadataReader

class RawDataSampler(MetadataReader):
    def __init__(self, data_dir: str):
        super().__init__(data_dir)

        self.flag_random_orientation = True
