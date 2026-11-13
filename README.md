## Monocular VSLAM: Feature-Based Monocular Visual SLAM
### Overview
This is a monocular visual SLAM system that builds a sparse 3D map of the environment using feature-based techniques. The system tracks camera motion and reconstructs environmental features using only visual input from a single camera.
### Key Features
- Camera tracking through feature matching and pose estimation
- Sparse 3D reconstruction of environmental features
- Feature extraction via ORB with multiscale pyramid support (learned extractors planned)
- Bundle adjustment for map and pose optimization (using g2o)
- Keyframe-based mapping
- Visualization using Open3D

## Setup and run
This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv, then run:

```bash
uv run python slam.py
```

On the first run, uv automatically downloads the pinned Python version (see `.python-version`), creates a virtual environment, and installs all dependencies.

**macOS note:** the Open3D wheel requires `libusb` at runtime. If `import open3d` fails with a dylib error, install it with Homebrew:

```bash
brew install libusb
```

## Custom keybindings for the renderer
- Press "q" to quit the application and store the session data
- Press "t" to toggle for automatic camera tracking
- Press "space" to pause/un-pause the SLAM process

## Configuration
Modify slam/config/config.yaml to customize:
```yaml
global:
  feature_extractor: "ORB"
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
3. Run ```uv run python slam.py```

## Dependencies
- OpenCV
- Open3D
- NumPy
- SciPy
- g2o (for bundle adjustment)
- Colorama (for logging)

Optional extras: `calib` (GeoCalib intrinsics priors), `learned` (PyTorch + torchvision — reserved for planned learned extractors/matchers).

## Future TODOs
- Improve keyframe selection logic. Its mostly governed by an interval currently since I haven't been able to get the more sophisticated selection process to work robustly.
- Figure out how to run the renderer in a separate thread. Due to GLFW limitations, the open3d renderer needs to run on the main thread. A way around this could be to run the SLAM pipeline in a separate worker thread instead, and let the renderer use the main thread.
- Add tooling for profiling
- Make it run faster...
- Add tooling for ground truth comparison for benchmarking
