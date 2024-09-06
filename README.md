## Sparse SLAM

### TODO:

    0. Frame processing module
    1. Feature detection
    2. Feature matching
    3. Estimate the essential matrix
    4. Estimate the pose of the camera
    5. Project image coordinates to 3d points
    6. Full bundle adjustment.
        - Research how this is used in context of SLAM (local vs global bundle adjustment)

### Future refactoring

Consider this structure when refactoring: 1. SLAM class that will have a constructure taking in the path for the video, and a run function that will run the algo. 2. Seperate utility functions into own file 3. Video processing, frame processing and feature extraction will go in its own processor module 4. Maybe keep matching in its own module as well 5. Create a MAP class that will contain necessary states about the map and camera poses over time.
