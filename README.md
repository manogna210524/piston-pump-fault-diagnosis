# Axial Piston Pump Fault Diagnosis

Condition monitoring for hydraulic axial piston pumps: detects internal leakage faults from the outlet pressure signal, so maintenance can target the exact faulty pistons.

From one pressure recording, the tool finds:

- whether the pump is **healthy or faulty**
- **how many** pistons are faulty
- **which** pistons are faulty and their **arrangement** (Inline / Random / Uniform)


## How it works

1. Simulate the pump in MATLAB Simscape for healthy and faulty cases.
2. Split the pressure signal into its rotation harmonics (FFT-based order tracking).
3. Estimate the leakage of each piston with least squares.
4. Match the faulty pistons to a known arrangement pattern.

## Results

Tested on 350+ simulated fault scenarios: 90% accuracy for fault detection, faulty piston count and arrangement.

