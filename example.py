"""Diagnose a single pressure record.

  python example.py --data data --file "data/300 bar_processed/N2_ARR03_RANDOM1/50um.csv"
"""
import argparse, json, os
import pandas as pd
from pump_diagnosis import Calibration, diagnose

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='data')
ap.add_argument('--file', required=True)
a = ap.parse_args()

rd = lambda f: pd.read_csv(f)['Net_Dis_pressure_Pa'].values
cal = Calibration(rd(os.path.join(a.data, 'Healthy.csv')),
                  rd(os.path.join(a.data, '300 bar_processed', 'N1_ARR01', '50um.csv')), 50, fs=1000.0, rpm=1470)
print(json.dumps(diagnose(rd(a.file), cal), indent=2, default=float))
