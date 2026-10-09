"""
Evaluate the diagnosis pipeline on the full simulated dataset.

Calibration uses only two files (Healthy.csv and N1_ARR01/50um.csv); every other file is test data.
Scenarios:
  A_full              full 10 s record
  B_random_crop       random 2-6 s segment, random start, shaft speed re-estimated from the signal
  C_crop+noise30kPa   B + 30 kPa white noise
  D_crop+noise100kPa  B + 100 kPa white noise

Usage:
  python evaluate.py --data data                 # all four scenarios
  python evaluate.py --data data --scenarios A_full
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
from pump_diagnosis import ARR, CANON, TYPE, Calibration, canon, diagnose

FS, RPM = 1000.0, 1470
SCENARIOS = ['A_full', 'B_random_crop', 'C_crop+noise30kPa', 'D_crop+noise100kPa']


def read_pressure(path):
    return pd.read_csv(path)['Net_Dis_pressure_Pa'].values


def list_files(data_dir):
    """(folder, arrangement number, clearance um, path) for the healthy file and every fault file."""
    fault_dir = os.path.join(data_dir, '300 bar_processed')
    files = [('Healthy', 0, 10, os.path.join(data_dir, 'Healthy.csv'))]
    for d in sorted(os.listdir(fault_dir)):
        m = re.match(r'N\d_ARR(\d+)', d)
        if not m:
            continue
        for f in sorted(glob.glob(os.path.join(fault_dir, d, '*um.csv'))):
            files.append((d, int(m.group(1)), int(os.path.basename(f)[:-6]), f))
    return files


def run(data_dir, scenarios, seed=7):
    fault_dir = os.path.join(data_dir, '300 bar_processed')
    cal = Calibration(read_pressure(os.path.join(data_dir, 'Healthy.csv')),
                      read_pressure(os.path.join(fault_dir, 'N1_ARR01', '50um.csv')), 50, FS, RPM)
    print(f'calibration: eps = {cal.eps:.5f}')
    files = list_files(data_dir)
    signals = {f: read_pressure(f) for *_, f in files}
    rng = np.random.default_rng(seed)
    rows = []
    for sc in scenarios:
        for folder, arr, h, f in files:
            p, t_first, refine = signals[f], 0.0, False
            if sc != 'A_full':
                i0 = int(rng.integers(500, 4000)); L = int(rng.integers(2000, 6000))
                p, t_first, refine = p[i0:i0 + L], i0 / FS, True
                if 'noise' in sc:
                    p = p + rng.normal(0, 30e3 if '30kPa' in sc else 100e3, len(p))
            r = diagnose(p, cal, t_first=t_first, refine_rpm=refine)
            S = ARR.get(arr, []) if h > 10 else []                 # 10 um == healthy clearance
            row = dict(scenario=sc, folder=folder, h=h, true_state='FAULTY' if S else 'HEALTHY', state=r['state'])
            if S:
                a_true = CANON[canon([int(k in S) for k in range(1, 10)])]
                row.update(N_true=len(S), N=r.get('N'), arr_true=a_true, arr=r.get('arrangement'),
                           type_true=TYPE[a_true], type=r.get('type'), h_est=r.get('severity_um', np.nan),
                           h_cls=r.get('severity_class_um'))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(d):
    out = {}
    for sc, g in d.groupby('scenario', sort=False):
        fa = g[g.true_state == 'FAULTY']
        err = (fa.h_est - fa.h).abs()
        out[sc] = {'detection': (g.state == g.true_state).mean(), 'N': (fa.N == fa.N_true).mean(),
                   'arrangement (34)': (fa.arr == fa.arr_true).mean(), 'type I/R/U': (fa.type == fa.type_true).mean(),
                   'severity (10 um class)': (fa.h_cls == fa.h).mean(), 'h MAE (um)': err.mean(), 'h max err (um)': err.max()}
    return pd.DataFrame(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data', help='folder containing Healthy.csv and "300 bar_processed/"')
    ap.add_argument('--scenarios', nargs='+', default=SCENARIOS, choices=SCENARIOS)
    ap.add_argument('--out', default='final_results.csv')
    a = ap.parse_args()
    d = run(a.data, a.scenarios)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    d.to_csv(a.out, index=False)
    print(summarize(d).round(3).to_string())
