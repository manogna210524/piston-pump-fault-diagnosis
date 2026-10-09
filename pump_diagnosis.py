"""
9-piston axial piston pump - leakage-fault diagnosis from the NET OUTLET PRESSURE signal only. No machine learning.

  Step 1  Healthy / Faulty      : relative mean-pressure drop + piston-imbalance energy vs healthy baseline
  Step 2  Number of faulty N    : order tracking -> piston-domain (9-point spatial DFT) deconvolution -> count
  Step 3  Arrangement           : rotation/reflection-invariant pattern of the faulty set -> 34 classes
                                  -> Inline / Random / Uniform
  Step 4  Severity h (um)       : laminar gap-leakage (h^3) + load-orifice (sqrt P) flow balance, inverted
                                  in closed form; split between pistons by the per-piston leakage map

Calibration (once per pump, not per fault): a healthy record and one single-piston-fault record
(gives the normalised leakage signature and the pump constant eps).  See Calibration below.
Requirement: a once-per-revolution key-phasor (tacho) reference. Pressure alone fixes the shaft angle only
modulo one piston pitch (40 deg), and the healthy inline pump is itself slightly asymmetric (manifold path
differences), so the healthy baseline must be subtracted at the true angle.
"""
import numpy as np

Z = 9                    # pistons
H0 = 10.0                # healthy clearance, um
N_ORD = 20               # shaft orders used (fs=1000 Hz, 24.5 Hz shaft -> Nyquist at 20.4 orders)
M = np.arange(N_ORD + 1)
MP = M[M % Z != 0]       # orders that are NOT multiples of 9: zero for a symmetric pump -> pure piston imbalance

# ------------------------------------------------------------------ arrangement taxonomy (34 classes)
ARR = {1:[1],2:[1,2],3:[1,3],4:[1,4],5:[1,5],6:[1,2,3],7:[1,2,4],8:[1,2,5],9:[1,3,5],10:[1,2,6],11:[1,4,7],
12:[1,2,3,4],13:[1,3,4,5],14:[1,4,5,6],15:[1,2,5,6],16:[1,2,4,5],17:[1,3,5,7],18:[1,2,3,4,5],19:[1,2,3,6,7],
20:[1,2,3,5,6],21:[1,2,3,4,6],22:[1,2,3,4,7],23:[1,2,4,6,8],24:[1,2,3,4,5,6],25:[1,2,3,5,6,7],26:[1,2,3,4,5,7],
27:[1,3,4,5,6,7],28:[1,3,5,6,7,8],29:[1,2,4,5,7,8],30:[1,2,3,4,5,6,7],31:[1,2,3,4,5,6,8],32:[1,2,3,4,5,7,8],
33:[1,2,3,4,6,7,8],34:list(range(1,9)),35:list(range(1,10))}      # ARR10 = P1,P2,P6 (as simulated)
INLINE = {1,2,6,12,18,24,30,34,35}; UNIFORM = {5,11,17,23,29,33}
TYPE = {a: 'INLINE' if a in INLINE else 'UNIFORM' if a in UNIFORM else 'RANDOM' for a in ARR}

def canon(bits):
    v = list(bits)
    return max(tuple(u[r:] + u[:r]) for u in (v, v[::-1]) for r in range(Z))
CANON = {}
for _a, _S in ARR.items():                       # ARR27 is the mirror image of ARR26 -> shares its class
    CANON.setdefault(canon([int(k in _S) for k in range(1, Z + 1)]), _a)

# ------------------------------------------------------------------ signal processing
K_ALIAS = 90             # also model shaft harmonics up to 90x (10 x piston-passing) as aliased nuisance terms

def order_spectrum(p, fs, f_shaft, t_first=0.0, skip=None):
    """Shaft-order amplitudes (orders 0..20) of ln(p) by least-squares harmonic regression on the shaft angle
    theta(t) = 2 pi f_shaft (t_first + n/fs).  t_first = time of the first sample after a key-phasor (tacho)
    pulse, so any record start / length gives the same angular reference.
    Harmonics above Nyquist (e.g. 27x, 36x, 45x of the piston ripple) fold back to known non-integer orders
    |m f_shaft - k fs|; they are included as nuisance regressors so they cannot leak into orders 0..20
    (aliasing-aware harmonic regression; equals the exact TSA/DFT for whole numbers of alias periods).
    skip = start-up transient [s]."""
    x = np.log(np.maximum(np.asarray(p, float), 1.0))
    i0 = int(round((skip if skip is not None else min(1.0, 0.2 * len(x) / fs)) * fs))
    x = x[i0:]
    t = (i0 + np.arange(len(x))) / fs
    th = 2 * np.pi * f_shaft * (t_first + t)
    cols = [np.ones_like(th)]
    for m in M[1:]:
        cols += [np.cos(m * th), np.sin(m * th)]
    # aliased harmonics: frequency folded into [0, fs/2]; phase referenced to the same shaft angle
    T = 1.0 / max(len(x) / fs, 1e-9); seen = [m * f_shaft for m in M]
    for m in range(N_ORD + 1, K_ALIAS + 1):
        fa = abs(((m * f_shaft + fs / 2) % fs) - fs / 2)
        if min(abs(fa - s) for s in seen) < 0.5 * T or fa < 0.5 * T or fs / 2 - fa < 0.5 * T:
            continue                                  # not resolvable from an existing term in this window
        seen.append(fa)
        cols += [np.cos(2 * np.pi * fa * t), np.sin(2 * np.pi * fa * t)]
    A = np.column_stack(cols)
    c = np.linalg.lstsq(A, x, rcond=None)[0]
    X = np.empty(len(M), complex); X[0] = c[0]
    X[1:] = c[1:2 * N_ORD + 1:2] - 1j * c[2:2 * N_ORD + 2:2]
    return X

def estimate_rpm(p, fs, rpm_nominal, span=0.03):
    """Refine shaft speed from the piston-passing (9x) ripple peak: zoom-DFT around 9 x nominal."""
    x = np.log(np.maximum(np.asarray(p, float), 1.0)); x = x[len(x) // 5:]; x = x - x.mean()
    f = np.linspace(1 - span, 1 + span, 3001) * Z * rpm_nominal / 60
    P = np.abs(np.exp(-2j * np.pi * np.outer(f, np.arange(len(x))) / fs) @ (x * np.hanning(len(x))))
    return f[int(np.argmax(P))] / Z * 60

def rotate(X, phi):            # shift the angular origin of an order spectrum by phi (rad)
    return X * np.exp(1j * M * phi)

def piston_map(D, G):
    """Per-piston leakage weights. Every piston produces the same signature G rotated by 40 deg:
         D_m = G_m * sum_k w_k exp(-i 2 pi m k / 9)      (9-point spatial DFT of w)
    Least squares over non-9x orders; their null-space (a common offset) is fixed by min(w)=0."""
    F = (G[:, None] * np.exp(-2j * np.pi * np.outer(M, np.arange(Z)) / Z))[MP]
    v = np.linalg.lstsq(np.vstack([F.real, F.imag]), -np.r_[D[MP].real, D[MP].imag], rcond=None)[0]
    return v - v.min()

def slot_to_piston(slot):      # slot 0 = P1 (template piston); P2, P3, ... fall at slots 8, 7, ...
    return (-slot) % Z + 1

# ------------------------------------------------------------------ severity physics
def sum_h3_ratio(x, eps):
    """Flow balance:  Q_th - L P = K sqrt(P),  L ~ sum h_k^3  (laminar gap leakage).
    With x = P/P_healthy and eps = healthy leakage fraction L0 P_h / Q_th:
         sum h_k^3 / (9 h0^3) = [ (1 - sqrt x)/eps + sqrt x ] / x"""
    return ((1 - np.sqrt(x)) / eps + np.sqrt(x)) / x

# ------------------------------------------------------------------ calibration + diagnosis
class Calibration:
    """Built once per pump from a healthy record and a single-piston fault record (any clearance).
    eps scales physically with operating point: eps ~ P_healthy / shaft_speed (leak ~ dp, Q_th ~ speed)."""
    def __init__(self, p_healthy, p_single, h_single, fs, rpm):
        self.fs, self.rpm = fs, rpm
        fr = rpm / 60
        self.Hb = order_spectrum(p_healthy, fs, fr)
        Xs = order_spectrum(p_single, fs, fr)
        g = Xs - self.Hb
        self.G = g / g[0]                                  # normalised single-piston signature
        x = np.exp(np.real(g[0]))
        # solve sum_h3_ratio(x, eps) = (8 h0^3 + h^3)/(9 h0^3) for eps
        target = (8 * H0 ** 3 + h_single ** 3) / (9 * H0 ** 3)
        self.eps = (1 - np.sqrt(x)) / (target * x - np.sqrt(x))
        self.P_h = np.exp(np.real(self.Hb[0]))
    def eps_at(self, P_h, rpm):
        return self.eps * (P_h / self.P_h) * (self.rpm / rpm)

DETECT_DROP = 5e-4    # 0.05 % relative mean drop  (smallest fault, 1 piston @ 20 um, gives 0.16 %)
R_ALL9 = 0.005        # imbalance ratio below this -> all 9 pistons leaking (N=9 <= 0.0014, N<=8 >= 0.0155)
REL_THR = 0.4         # piston faulty if w > 0.4 * max(w)  (faulty >= 0.65, healthy <= 0.20 observed)

def diagnose(p, cal, fs=None, rpm=None, t_first=0.0, Hb=None, P_h=None, refine_rpm=False):
    """p: net outlet pressure samples (Pa); t_first: time of first sample after the key-phasor pulse.
    Hb/P_h: healthy baseline at this operating point (defaults to the calibration point)."""
    fs = fs or cal.fs; rpm = rpm or cal.rpm
    if refine_rpm:
        rpm = estimate_rpm(p, fs, rpm)
    Hb = cal.Hb if Hb is None else Hb
    X = order_spectrum(p, fs, rpm / 60, t_first)
    D = X - Hb
    drop = -np.real(D[0])                                   # relative mean-pressure drop
    out = dict(mean_drop_pct=100 * (1 - np.exp(-drop)))
    # ---- Step 1
    if drop < DETECT_DROP:
        out.update(state='HEALTHY'); return out
    out['state'] = 'FAULTY'
    # ---- Step 2 & 3
    R = np.linalg.norm(D[MP]) / abs(D[0])
    if R < R_ALL9:
        w = np.ones(Z); bits = np.ones(Z, int)
    else:
        w = piston_map(D, cal.G); w = w / w.max(); bits = (w > REL_THR).astype(int)
    a = CANON.get(canon(bits))
    out.update(N=int(bits.sum()), arrangement=a, type=TYPE.get(a, 'NOT-PREDEFINED'), imbalance_R=R,
               faulty_pistons=sorted(slot_to_piston(s) for s in np.flatnonzero(bits)),
               piston_weights={slot_to_piston(s): round(float(w[s]), 3) for s in range(Z)})
    # ---- Step 4
    eps = cal.eps_at(P_h or np.exp(np.real(Hb[0])), rpm)
    excess = 9 * H0 ** 3 * (sum_h3_ratio(np.exp(-drop), eps) - 1)        # sum_k (h_k^3 - h0^3)
    wf = np.where(bits, w, 0.0)
    hk = np.cbrt(H0 ** 3 + excess * wf / wf.sum())                      # per-piston clearance
    h_eq = float(np.cbrt(H0 ** 3 + excess / bits.sum()))                # equal-clearance estimate
    out.update(severity_um=round(h_eq, 1), severity_class_um=int(min(max(10 * round(h_eq / 10), 20), 100)),
               per_piston_clearance_um={slot_to_piston(s): round(float(hk[s]), 1) for s in range(Z)})
    return out
