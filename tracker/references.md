# ByteTrack Implementation Reference

Detailed references to external repositories used in this application development.

## References

- **ByteTrack** (MIT License, Date: 31 July, 2023)

  - **Repository:** [https://github.com/ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack/tree/main/yolox/tracker)
  - **Folder Used:** [https://github.com/ifzhang/ByteTrack/tree/main/yolox/tracker](https://github.com/ifzhang/ByteTrack/tree/main/yolox/tracker)
  - **Original Author:** [Yifu Zhang](https://github.com/ifzhang)
  - **Description:** This repository includes the ByteTrack implementation using YOLOX. ByteTrack is a state-of-the-art object tracking algorithm that we have integrated into our project to enhance its object tracking capabilities. We have used this with YOLOv7 integration.
- **cython_bbox** - A single function bbox_overlaps (MIT License, Date: 31 July, 2023)

  - **Repository:** [https://github.com/samson-wang/cython_bbox/tree/master](https://github.com/samson-wang/cython_bbox/blob/17a9209162d2107c1d2d17ea66fb2a3c7fa0c73e/src/cython_bbox.pyx#L15C1-L55C20)
  - **Function Used:** [https://github.com/samson-wang/cython_bbox/blob/17a9209162d2107c1d2d17ea66fb2a3c7fa0c73e/src/cython_bbox.pyx#L15C1-L55C20](https://github.com/samson-wang/cython_bbox/blob/17a9209162d2107c1d2d17ea66fb2a3c7fa0c73e/src/cython_bbox.pyx#L15C1-L55C20)
  - **Original Author:** [samson-wang](https://github.com/samson-wang)
  - **Description:** This repository provides the function for plotting bounding boxes with memory optimization etc. We have taken a **bbox_overlaps** function from this repository and used in our exisiting application implementation.

Some changes have been incorporated into the relevant sources above when including in this repo.