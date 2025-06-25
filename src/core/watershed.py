import logging

from .base import BaseProcessor
from ..data_io import load_static_data

logger = logging.getLogger(__name__)

class WatershedLoader(BaseProcessor):

    def load(self, var_name):
        """Load watershed data from GeoTIFF."""
        return load_static_data(self.config, var_name = var_name)