"""PanSubNet - cell + patch cross-attention MIL for pancreatic cancer WSI classification."""

from pansubnet import utils
from pansubnet.dataset import WSIDataset
from pansubnet.model import WSIClassifier

__all__ = ["utils", "WSIDataset", "WSIClassifier"]
__version__ = "0.1.0"
