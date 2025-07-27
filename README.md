## Monocular VSLAM: Feature-Based Monocular Visual SLAM
### Overview
This is a monocular visual SLAM system that builds a sparse 3D map of the environment using feature-based techniques. The system tracks camera motion and reconstructs environmental features using only visual input from a single camera.
### Key Features
- Real-time camera tracking using feature matching and pose estimation
- Sparse 3D reconstruction of environmental features
- Multiple feature extractors supported (ORB, AKAZE, SuperPoint)
- Bundle adjustment for map and pose optimization (using g2o)
- Keyframe-based mapping
- Visualization using Open3D

### Example run using A-KAZE for feature extraction:

https://github.com/user-attachments/assets/ff7c8afa-c63a-49e7-aa66-5c6ffd1ad6b9



### Example run using SuperPoint on [TUM rgbd](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) dataset (fr1/xyz)
This is using the SuperPoint feature extractor. Generates tons of good keypoints and its able to produce fairly dense maps, but its quite costly to run.

https://github.com/user-attachments/assets/cb612c58-9a4e-4216-81dc-7656cf5de836


## Setup and run
Install the required dependencies, ideally in a new environment:

```bash
pip install -r requirements.txt
```

The weights for SuperPoint are already provided in the repo.

## Configuration
Modify slam/config/config.yaml to customize:
```yaml
global:
  feature_extractor: "DNN"  # ORB, AKAZE, or DNN
  enable_multiscale_features: true
  record_session: true

videos:
  tum_xyz:  # Example dataset configuration
    path: "data/rgbd_dataset_freiburg1_xyz/"
    fx: 517.3
    fy: 516.5
    cx: 318.6
    cy: 255.3
```

## Example: TUM RGBD Dataset
1. Download the [TUM RGBD dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
2. Configure the dataset path in ```config.yaml```
3. Run ```python slam.py```

## Dependencies
- OpenCV
- Open3D
- NumPy
- SciPy
- PyTorch (for SuperPoint)
- g2o (for bundle adjustment)
- Colorama (for logging)

## Future TODOs
- Add loop closure detection
- Add relocalization strategy for when we lose track
- Improve keyframe selection criteria. Its mostly governed by an interval currently since I haven't been able to get the more sophisticated selection process to work robustly.
- Add tooling for profiling
- Improve efficiency, its barely usable for real-time purposes currently.
- Add tooling for ground truth comparison for benchmarking
