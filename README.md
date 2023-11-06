# RGB-D Mapping and Tracking in a Plenoxel Radiance Field

## Setup

First create the virtualenv; we recommend using conda:
```sh
conda env create -f environment.yml
conda activate plemt
```

If your CUDA toolkit is older than 11, then you will need to install CUB as follows:
`conda install -c bottler nvidiacub`.
Since CUDA 11, CUB is shipped with the toolkit.

To install the main library (svox2), simply run
```
pip install .
```
In the repo root directory.


## Mapping

To build a map for a scene, use `opt/opt_rgbd.py`.
```
python opt/opt_d.py [dataset type: scan/rplc] [sequence dir] -t ckpt
```

You can compare the results from opt/opt.py, which focuses solely on optimizing the radiance field using RGB information. The usage is the same as demonstrated above.
After running, you can find built map at `ckpt/ckpt.npz`.


## Tracking
For tracking, yaml file of specific sequence inside `/opt/configs/config_for_tracking` folder should be adjusted.
Change `ckpt` and `data_path` properly.

after modifying yaml file, you can run tracking code like this:
```
python opt/run.py 'opt/configs/config_for_tracking/replica/office_0.yaml'
```

## Evaluation
before running evaluation code, you should modify the line in `evo` package.
inside `/root/anaconda3/envs/plemt/lib/python3.8/site-packages/evo/core/lie_algebra.py`, line 186, find this part:

```python
def is_so3(r: np.ndarray) -> bool:
    """
    :param r: a 3x3 matrix
    :return: True if r is in the SO(3) group
    """
    # Check the determinant.
    det_valid = np.allclose(np.linalg.det(r), [1.0], atol=1e-6)
    # Check if the transpose is the inverse.
    inv_valid = np.allclose(r.transpose().dot(r), np.eye(3), atol=1e-6)
    return det_valid and inv_valid
```
and change threshold from 1e-6 to 1e-5 to avoid numerical problems.

By running this: 
```
python opt/scoring.py --path=[logs/replica/office_0]
```
You can obtain results.csv that reports ATE, RPE_t, RPE_r, and averate tracking time.

To see the rendered RGB image or RGB-D image from ckpt, 
```
python opt/render_imgs.py [CHECKPOINT.npz] [dataset type: scan/rplc] [sequence dir]
python opt/render_imgs_d.py [CHECKPOINT.npz] [dataset type: scan/rplc] [sequence dir]
``````

This code is highly based on the repository below:
# Plenoxels: Radiance Fields without Neural Networks

Alex Yu\*, Sara Fridovich-Keil\*, Matthew Tancik, Qinhong Chen, Benjamin Recht, Angjoo Kanazawa
UC Berkeley

Website and video: <https://alexyu.net/plenoxels>
arXiv: <https://arxiv.org/abs/2112.05131>