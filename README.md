# GPU-Accelerated Chirplet Transform
Hardware-accelerated implementation of the Adaptive Chirplet Transform (ACT) for signal processing tasks. Designed for fast and efficient feature extraction.

## 📁 File Structure

- `act_gpu_faster_dict_gen.py`: Main ACT class with GPU-accelerated dictionary generation and transform.
- `monitoringclass.py`: Monitors CPU and GPU usage during execution.
- `run_chirplet_transform.py`: Example script to apply the transform on EEG data.
- `ACT/Bitbrain/sub-1/...`: Directory containing EEG `.edf` files for input.
- `act_results_sub-1_optimized.csv`: Output file containing per-epoch transform results.

## 🔧 Dependencies

Make sure you have the following installed:

```bash
pip install cupy mne numpy pandas scipy matplotlib tqdm joblib
```

Other requirements:

- **CUDA-compatible GPU**
- **Python 3.8+**
- NVIDIA driver and CUDA toolkit (>= 12.0)

## 💡 Usage

### 1. Prepare EEG Data

Ensure your `.edf` EEG file is located at:

```
ACT/Bitbrain/sub-1/eeg/sub-1_task-Sleep_acq-headband_eeg.edf
```

You may modify the path in `run_chirplet_transform.py` if needed.

### 2. Run Transform

```bash
python run_chirplet_transform.py
```

This will:

- Load EEG data and filter it
- Run the chirplet transform on `HB_1` and `HB_2` channels
- Save the chirplet parameters, coefficients, error, and residue to `act_results_sub-1_optimized.csv`

### 3. Output Format

The output CSV contains:

| Epoch | Params (tc, fc, logDt, c) | Coeffs | Error | Residue |
|-------|----------------------------|--------|-------|---------|

## 🧠 Example: Chirplet Atom Generation

You can generate a single chirplet atom using:

```python
act = ACT(FS=256, length=1280)
chirplet_atom = act.g(tc=640, fc=5, logDt=-2, c=0.5)
```

## 📈 Performance Monitoring

CPU and GPU usage are tracked per epoch using the `MonitoringClass`.

---

## 📜 License

MIT License. See `LICENSE` file for details.

## 👨‍🔬 Author

Nish K.  
Contact: [your_email@example.com]

---

## 📌 Notes

- This implementation uses **CuPy's Unified Memory** to support large-scale signal processing with minimal memory management overhead.
- The transform uses `scipy.optimize.minimize` to refine chirplet parameters after dictionary matching.
- Suitable for sparse signal decomposition and high-resolution time-frequency analysis.

---

## ✅ To Do

- [ ] Add visualization of decomposition results
- [ ] Benchmark dictionary generation time vs CPU
- [ ] Enable batch processing of multiple EEG files
