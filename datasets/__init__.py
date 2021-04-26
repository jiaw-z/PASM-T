from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDataset

__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset
}
