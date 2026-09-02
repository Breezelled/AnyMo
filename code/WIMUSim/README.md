# WIMUSim: Wearable IMU Simulation Framework

WIMUSim models wearable IMU data using four key parameters: \textit{Body (B)}, \textit{Dynamics (D)}, \textit{Placement (P)}, and \textit{Hardware (H)}. These parameters are designed to reflect the real-world variabilities that affect wearable IMU data, offering an intuitive and physically plausible wearable IMU simulation framework.

- **Body (B)** defines the structural characteristics of the human body model, specifying the length of each limb to construct a skeletal representation as a directed tree. These measurements can be manually entered or derived from anthropometric databases \citep{gordon20142012, openerg} for default values.
- **Dynamics (D)** represents the temporal sequence of movements using rotation quaternions for each joint, depicting their orientation over time relative to parent joints, alongside a sequence of 3D vectors for overall body translation. This can be extracted from motion capture data, whether sourced from IMUs or optical systems or analyzed from video sequences.
- **Placement (P)** specifies the position and orientation of the virtual IMUs relative to their associated body joints. This parameter is specified manually based on expected sensor placement in the target environment, but it may also be varied to simulate different sensor placement scenarios.
- **Hardware (H)** models each IMU's specific operational characteristics, such as sensor biases and noise levels. By incorporating these parameters, WIMUSim ensures that the generated virtual IMU data accurately reflects real-world IMU performances. These parameters can be manually specified based on device specifications.

WIMUSim is designed to be used as follows:
- **Data Collection**: Prepare real IMU data and preliminary WIMUSim parameters. The \textit{B} and \textit{D} can be derived from various motion capture technologies, including optical-based, IMU-based, or video-based techniques. The \textit{P} is manually specified to indicate where the real IMU is placed. The \textit{H} can be obtained from device specifications or data collected at stationary positions. At this point, these parameters can be rough estimates.
- **High Fidelity Parameter Identification**: Optimize these preliminary parameters by minimizing the error between the real and virtual IMU data to ensure that WIMUSim accurately parametrizes the real IMU data. This optimization is performed using a gradient descent-based method to minimize the error between the real and virtual IMU data. (see examples/parameter_identification.ipynb)
- **Realistic Transformation**: Adjust the parameters around their identified operating points to introduce physically plausible variabilities, generating virtual IMU data that reflects a broad range of realistic conditions. This allows for the creation of diverse and enriched datasets, enhancing the training of HAR models without requiring extensive new data collection. (see examples/parameter_transformation.ipynb)


## Getting Started

### Requirements
- Python 3.8 or higher
- torch (>=2.0)
- pytorch3d (>=0.7.5)
See notebook examples in the `examples` folder for a detailed guide on how to use WIMUSim.


## Installation (if you want to use WIMUSim in your own project)
```bash
git clone https://github.com/STRCWearlab/WIMUSim.git
pip install --dev WIMUSim
```


## Citation


## License
Apache License 2.0

