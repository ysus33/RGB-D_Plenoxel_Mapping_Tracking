from .replica_dataset import ReplicaDataloader
from .scannet_dataset import ScanNetDataloader

datasets = {
    'replica': ReplicaDataloader,
    'scannet': ScanNetDataloader,
}