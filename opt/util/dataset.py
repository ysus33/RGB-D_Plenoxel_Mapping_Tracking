from .rplc_dataset import RplcDataset
from .scan_dataset import SCANDataset
from os import path

datasets = {
    'rplc': RplcDataset,
    'scan': SCANDataset
}
