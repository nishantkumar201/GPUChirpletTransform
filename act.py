import os
import numpy as np
import scipy.optimize as optimize
import joblib
import psutil

class ACT:
    def __init__(
        self,
        FS=256,
        length=3840,
        dict_addr="dict_cache.p",
        tc_info=(0, 3840, 1),
        fc_info=(0.7, 15, 0.2),
        logDt_info=(-4, -1, 0.3),
        c_info=(-30, 30, 3),
        complex=False,
        force_regenerate=False,
        mute=False,
    ):
        self.FS = FS
        self.length = length
        self.dict_addr = dict_addr

        self.tc_info = tc_info
        self.fc_info = fc_info
        self.logDt_info = logDt_info
        self.c_info = c_info

        self.complex = complex
        self.float32 = True

        if not mute:
            print("\n=== INITIALIZING ADAPTIVE CHIRPLET TRANSFORM MODULE ===\n")

        if os.path.exists(self.dict_addr) and not force_regenerate:
            if not mute:
                print("Found cached chirplet dictionary. Loading...")
            self.dict_mat, self.param_mat = joblib.load(self.dict_addr)
        else:
            if not mute:
                print("Generating chirplet dictionary...")
            self.generate_chirplet_dictionary(debug=True)
            if not mute:
                print("Caching dictionary...")
            joblib.dump((self.dict_mat, self.param_mat), self.dict_addr)
            if not mute:
                print("Done.")

        if not mute:
            print("=== DONE INITIALIZING ACT MODULE ===\n")

    def g(self, tc=0, fc=1, logDt=0, c=0):
        """
        Generate a Gaussian chirplet (unit-energy normalized).
        """
        tc /= self.FS  # convert to seconds
        Dt = np.exp(logDt)
        t = np.arange(self.length) / self.FS

        gaussian_window = np.exp(-0.5 * ((t - tc) / Dt) ** 2)
        complex_exp = np.exp(2j * np.pi * (c * (t - tc) ** 2 + fc * (t - tc)))

        chirplet = gaussian_window * complex_exp
        if not self.complex:
            chirplet = np.real(chirplet)

        # UNIT-ENERGY NORMALIZATION
        norm = np.linalg.norm(chirplet)
        if norm > 0:
            chirplet /= norm

        if self.float32:
            chirplet = chirplet.astype(np.float32)

        return chirplet

    def generate_chirplet_dictionary(self, debug=False):
        tc_vals = np.arange(*self.tc_info)
        fc_vals = np.arange(*self.fc_info)
        logDt_vals = np.arange(*self.logDt_info)
        c_vals = np.arange(*self.c_info)

        dict_size = len(tc_vals) * len(fc_vals) * len(logDt_vals) * len(c_vals)
        if debug:
            print(f"Dictionary length: {dict_size}")

        dict_mat = np.zeros([dict_size, self.length], dtype=np.float32)
        param_mat = np.zeros([dict_size, 4], dtype=np.float32)

        cnt = 0
        for tc in tc_vals:
            for fc in fc_vals:
                for logDt in logDt_vals:
                    for c in c_vals:
                        dict_mat[cnt] = self.g(tc, fc, logDt, c)
                        param_mat[cnt] = [tc, fc, logDt, c]
                        cnt += 1

        print("CPU usage during dictionary generation:", psutil.cpu_percent(0))
        self.dict_mat = dict_mat
        self.param_mat = param_mat
        return dict_mat, param_mat

    def search_dictionary(self, signal):
        """
        Find dictionary atom with maximum correlation to the signal.
        """
        projections = self.dict_mat.dot(signal)
        ind = np.argmax(np.abs(projections))
        return ind, projections[ind]

    def minimize_this(self, coeffs, signal):
        """
        BFGS cost function: maximize correlation with residual.
        """
        atom = self.g(*coeffs)
        return -1.0 * abs(atom.dot(signal))  # maximize inner product

    def transform(self, signal, order=5, debug=True):
        param_list = np.zeros([order, 4], dtype=np.float32)
        coeff_list = np.zeros(order, dtype=np.float32)
        approx = np.zeros(len(signal), dtype=np.float32)
        residue = np.copy(signal)

        if debug:
            print(f"Beginning {order}-order ACT transform...")

        for P in range(order):
            if debug:
                print(f"Processing atom {P+1}/{order}...")

            # 1) Find best matching atom
            ind, _ = self.search_dictionary(residue)
            params = self.param_mat[ind]

            # 2) Refine using optimizer
            res = optimize.minimize(
                self.minimize_this, params, args=(residue,), method="BFGS"
            )
            new_params = res.x
            if res.status != 0 and debug:
                print(f"Optimizer did not converge: {res.message}")

            # 3) Generate refined atom (unit-energy)
            updated_chirp = self.g(*new_params)

            # 4) Compute coefficient using residual
            coeff = updated_chirp.dot(residue)

            # 5) Update residual and approximation
            residue -= updated_chirp * coeff
            approx += updated_chirp * coeff

            # 6) Store parameters and coefficient
            param_list[P] = new_params
            coeff_list[P] = coeff

        return {
            "params": param_list,
            "coeffs": coeff_list,
            "signal": signal,
            "error": np.sum(residue),
            "residue": residue,
            "approx": approx,
        }
