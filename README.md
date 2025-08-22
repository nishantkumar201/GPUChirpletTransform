# Adaptive Chirplet Transform (ACT) – GPU Implementation

This repository provides a **GPU-accelerated implementation of the Adaptive Chirplet Transform (ACT)** for EEG analysis.  
It leverages **NVIDIA CUDA (via [CuPy](https://cupy.dev))** to accelerate dictionary generation and iterative decomposition of EEG signals, with optional CPU/GPU monitoring utilities.

The repository currently includes three core modules:

- **`act_gpu.py`** – Implements the Adaptive Chirplet Transform, including GPU-accelerated dictionary generation and signal decomposition.  
- **`monitoringclass.py`** – Provides CPU/GPU usage monitoring (power, memory, utilization), logging results to CSV for later inspection.  
- **`run_act_example.py`** – Example pipeline for loading EEG data, preprocessing, applying ACT, and exporting results.  

---
## Features

- GPU-accelerated chirplet dictionary construction using CuPy  
- Iterative ACT decomposition with parameter refinement via SciPy optimization  
- EEG preprocessing supported through [MNE](https://mne.tools/)  
- Optional CPU/GPU monitoring during runs  
- Outputs results (parameters, coefficients, reconstruction errors) as CSV  

---

## Requirements

- **Python** 3.9+  
- **NVIDIA GPU** with CUDA support (tested with CUDA 12.x)  
- Recommended: 8 GB+ VRAM for larger EEG datasets  

### Python Dependencies
All required packages are listed in `requirements.txt`.  
Install them with:
~~~bash
pip install -r requirements.txt
~~~

## Quick Start

This guide will help you set up and run the provided code examples quickly.

---

### 1. Clone the Repository

~~~bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
~~~

### 2.	Prepare EEG data
Place .edf EEG files in the expected directory structure.
Example path used in run_act_example.py:

~~~
ACT/Bitbrain/sub-1/eeg/sub-1_task-Sleep_acq-headband_eeg.edf
~~~

### 3. Run the example script

~~~
python run_act_example.py
~~~

### 	4.	Check outputs
Result Logs:
~~~
act_results_sub-1_optimized.csv
~~~
Monitoring logs:
~~~
cpu_monitoring.csv
gpu_monitoring.csv
~~~

## Output Format

The output CSV contains:

| Epoch | Params (tc, fc, logDt, c) | Coeffs | Error | Residue |
|-------|----------------------------|--------|-------|---------|

## Repository Structure
~~~
├── act_gpu.py             # Chirplet dictionary & transform (GPU accelerated)
├── monitoringclass.py     # CPU/GPU monitoring utilities
├── run_act_example.py     # Example script for EEG processing
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── LICENSE                # MIT license (recommended)
~~~

## Citation
If you use this code in your research, please cite the upcoming preprint (link will be added when available).

## Author
Nishant Kumar

## License
This project is released under the MIT License.


