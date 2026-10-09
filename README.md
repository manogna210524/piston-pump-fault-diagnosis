# Axial Piston Pump Fault Diagnosis

Condition monitoring for hydraulic axial piston pumps: detects internal leakage faults from the outlet pressure signal, so maintenance can target the exact faulty pistons.

From one pressure recording, the tool finds:

- whether the pump is **healthy or faulty**
- **how many** pistons are faulty
- **which** pistons are faulty and their **arrangement** (Inline / Random / Uniform)

![Piston leakage maps](docs/figures/f3_piston_maps.png)

## How it works

1. Simulate the pump in MATLAB Simscape for healthy and faulty cases.
2. Split the pressure signal into its rotation harmonics (FFT-based order tracking).
3. Estimate the leakage of each piston with least squares.
4. Match the faulty pistons to a known arrangement pattern.

## Results

Tested on 350+ simulated fault scenarios: 100% accuracy for fault detection, faulty piston count and arrangement.

## Files

| File | Purpose |
|---|---|
| `pump_diagnosis.py` | Diagnosis method |
| `example.py` | Diagnose a single recording |
| `evaluate.py` | Run the full evaluation |

## Run

```bash
pip install -r requirements.txt
python example.py --data data --file "data/300 bar_processed/N2_ARR03_RANDOM1/50um.csv"
```

The simulation data is not included in this repository.
