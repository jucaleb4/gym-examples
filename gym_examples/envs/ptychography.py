import gymnasium as gym
from gymnasium import spaces

import numpy as np
import numpy.linalg as la
from skimage.transform import radon, iradon_sart

from . import PhantomGenerator, utils, ptychography_helper
from .PhantomGenerator import PhantomGenerator
from .utils import ImageType, RewardType

import math
from scipy.optimize import minimize

from .ptychography_helper import (
    _idx, coverage, extract, scatter, forward, measure, gaussian_probe, align,
)

try:
    from utils import ImageType, RewardType
except Exception:                                            # pragma: no cover
    from enum import Enum

    class ImageType(Enum):
        MIXED = 1; TRIANGLE = 2; TRIANGLE_THIRTY = 3; TRIANGLE_FIXED = 4
        LINE = 5; LINE_THIRTY = 6; LINE_FIXED = 7

    class RewardType(Enum):
        PNSR_E2E = 1; PNSR_INCREMENTAL = 2; PNSR_ADD = 3
        FWD_E2E = 4; FWD_INCREMENTAL = 5
        FWD_RND_1 = 6; FWD_RND_10 = 7
        FWD_RND_COMPLETE_10 = 8; FWD_RND_COMPLETE_FULL = 9
        FWD_INC_RND_1 = 10; FWD_INC_RND_10 = 11
        FWD_RND_1_REG = 12

try:
    from PhantomGenerator import PhantomGenerator
except Exception:                                            # pragma: no cover
    PhantomGenerator = None

# https://stackoverflow.com/questions/24825441/stop-scipy-minimize-after-set-time
import warnings, time
class TookTooLong(Warning):
    pass

class MinimizeStopper(object):
    def __init__(self, max_sec=300):
        self.max_sec = max_sec
        self.start = time.time()
    def __call__(self, xk=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)

# ======================================================================
# 1. penalty functions c(a, b),  a = |F(w .* S_j z)|^2  (model intensity)
#                                b = d_j                 (measured intensity)
#    each returns (value, dc/da).  dc/da is all the solver needs.
# ======================================================================

_EPS = 1e-12


def _c_intensity(a, b):
    """Gaussian / least-squares on intensities: (a - b)^2."""
    r = a - b
    return r ** 2, 2.0 * r


def _c_amplitude(a, b):
    """Amplitude least squares: (sqrt(a) - sqrt(b))^2.  Best conditioned."""
    sa, sb = np.sqrt(a + _EPS), np.sqrt(b)
    return (sa - sb) ** 2, 1.0 - sb / sa


def _c_poisson(a, b):
    """Poisson / shot-noise MLE (Kullback-Leibler): a - b log a."""
    aa = a + _EPS
    return aa - b * np.log(aa), 1.0 - b / aa


def _c_huber(a, b, delta=1.0):
    """Huber on the amplitude residual -- robust to outlier pixels."""
    r = np.sqrt(a + _EPS) - np.sqrt(b)
    big = np.abs(r) > delta
    val = np.where(big, delta * (2 * np.abs(r) - delta), r ** 2)
    dcdr = np.where(big, 2 * delta * np.sign(r), 2 * r)
    return val, dcdr * 0.5 / np.sqrt(a + _EPS)


PENALTIES = {
    "intensity": _c_intensity,
    "amplitude": _c_amplitude,
    "poisson":   _c_poisson,
    "huber":     _c_huber,
}


# ======================================================================
# 2. objective + analytic gradient, packed into a real vector for scipy
# ======================================================================

def _pack(z, w):
    return np.concatenate([z.real.ravel(), z.imag.ravel(),
                           w.real.ravel(), w.imag.ravel()])


def _unpack(x, n, m):
    k = n * n
    z = (x[:k] + 1j * x[k:2 * k]).reshape(n, n)
    w = (x[2 * k:2 * k + m * m] + 1j * x[2 * k + m * m:]).reshape(m, m)
    return z, w


def objective(x, d2, pos, n, m, penalty="amplitude", periodic=True,
              fix_probe=None, reg=0.0, normalize=True):
    """f(z, w) = sum_j c(|F(w .* S_j z)|^2, d_j)   and its real gradient.

    Wirtinger calculus:  with y = F(w .* S_j z) and g_j = F^H( c_a(|y|^2) * y ),
        df/d(conj z) = sum_j S_j^T ( conj(w) * g_j )          <- scatter
        df/d(conj w) = sum_j conj(S_j z) * g_j                <- sum over j
    and the gradient w.r.t. the stacked real vector is 2 * df/d(conj .).
    """
    z, w = _unpack(x, n, m)
    if fix_probe is not None:
        w = fix_probe

    psi = extract(z, pos, m, periodic) * w[None]           # w .* S_j z
    y = np.fft.fft2(psi, norm="ortho")
    a = np.abs(y) ** 2

    cval, ca = PENALTIES[penalty](a, d2)
    f = float(cval.sum())

    g = np.fft.ifft2(ca * y, norm="ortho")                 # F^H (c_a * y)
    gz = scatter(np.conj(w)[None] * g, pos, n, periodic)
    gw = np.sum(np.conj(extract(z, pos, m, periodic)) * g, axis=0)

    if reg:                                                # Tikhonov, keeps the
        f += reg * (np.vdot(z, z).real + np.vdot(w, w).real)   # never-scanned
        gz = gz + reg * z                                      # pixels bounded
        gw = gw + reg * w

    if fix_probe is not None:
        gw = np.zeros_like(gw)

    scale = 1.0 / max(d2.size, 1) if normalize else 1.0
    grad = 2.0 * _pack(gz, gw) * scale
    return f * scale, grad


# ======================================================================
# 3. generic scipy solver wrapper -- swap `method` freely
# ======================================================================

# methods that accept an analytic jacobian and scale to 1e4+ variables
SCIPY_METHODS = ("L-BFGS-B", "CG", "BFGS", "TNC", "Newton-CG", "trust-constr", "Powell")


def solve_blind(d, pos, n, m, z0=None, w0=None, penalty="amplitude",
                method="L-BFGS-B", maxiter=200, periodic=True, reg=1e-8,
                fix_probe=False, options=None, callback=None):
    """min_{w,z} sum_j c(|F(w .* S_j z)|^2, d_j) with scipy.optimize.minimize.

    d : (J, m, m) measured *magnitudes* (as returned by ptychography.measure)
    returns (z_hat, w_hat, scipy OptimizeResult)
    """
    d2 = d.astype(float) ** 2                              # intensities
    if z0 is None:
        z0 = np.ones((n, n), dtype=complex)
    if w0 is None:
        w0 = gaussian_probe(m, 0.35, 0.0).astype(complex)

    probe = w0.copy() if fix_probe else None
    args = (d2, np.asarray(pos, int), n, m, penalty, periodic, probe, reg)

    # scipy's default ftol stops L-BFGS-B absurdly early on this objective
    opt = {"maxiter": maxiter}
    if method in ("L-BFGS-B", "TNC"):
        opt.update(maxfun=10 * maxiter, ftol=1e-6, gtol=1e-6)
    elif method in ("CG", "BFGS", "Newton-CG"):
        opt["gtol"] = 1e-6
    if options:
        opt.update(options)

    callback=MinimizeStopper()
    res = minimize(objective, _pack(z0, w0), args=args, jac=True,
                   method=method, options=opt, callback=callback)

    z, w = _unpack(res.x, n, m)
    if fix_probe:
        w = w0
    else:                                                  # fix the (cz, w/c)
        s = la.norm(w0) / (la.norm(w) + _EPS)              # scale ambiguity
        w, z = w * s, z / s
    return z, w, res


# ======================================================================
# 4. action space: the candidate scanning windows S_j
# ======================================================================

def psnr(target, ref):
    diff = (ref - target).ravel()
    rmse = math.sqrt(np.mean(np.abs(diff) ** 2))
    return 20 * math.log10(1.0 / max(rmse, _EPS))

class PtychographyEnv(gym.Env):
    def __init__(
        self, seed, n_images, num_scans, image_size, action_size,
        probe_size, corruption_photons, pnsr_reg,
        image_type, reward_type,
        penalty, solver, solver_iters,
        periodic, jitter, allow_repeats,
        state_mode, warm_start, blind,
    ):
        self.image_size = 256 # image_size            # n
        self.probe_size = probe_size            # m
        self.num_scans = num_scans              # N = episode length
        self.action_size = action_size
        self.periodic = periodic
        self.penalty = penalty
        self.solver = solver
        self.solver_iters = solver_iters
        self.reward_type = reward_type
        self.pnsr_reg = pnsr_reg
        self.allow_repeats = allow_repeats
        self.state_mode = state_mode
        self.warm_start = warm_start
        self.blind = blind                      # False -> probe known (easier)
        self.corruption_photons = corruption_photons
        self.init_start = 0
        self.img_channels = 1 + (self.state_mode != "amp") # only amp has 1 channels

        # --- objects of interest (complex: amplitude + phase from a phantom)
        self.P_all = self._make_objects(seed, n_images, image_size, image_type)
        self.w_true = gaussian_probe(probe_size)

        # --- the action space: one S_j per action
        self.positions = self.candidate_windows(image_size, probe_size, action_size,
                                           periodic, jitter, seed)

        self.n = self.np_random.choice(len(self.P_all))

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.img_channels, self.image_size, self.image_size), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(action_size)

    def _get_obs(self):
        return np.array(self.state)

    def _get_info(self):
        return {"image_id": self.n}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.n = int(self.np_random.choice(len(self.P_all)))
        self.a_start = self.init_start
        self.curr_iteration = 0
        self.pos_seq = []                      # list of chosen S_j corners
        self.action_seq = []
        self.d_seq = []                        # list of (m,m) magnitude frames
        self.previous_reward = 0.0
        self.reward = 0.0
        self.done = False
        self.z_hat = np.ones((self.image_size,) * 2, dtype=complex)
        self.w_hat = gaussian_probe(self.probe_size, 0.35, 0.0).astype(complex)
        self.state = self._state_from(self.z_hat * 0)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        action = int(action)
        self.a_start += 1
        self.action_seq.append(action)
        self.window_action = self.positions[action]
        self.pos_seq.append(self.window_action)

        # --- acquire the phaseless frame d_j for the chosen window
        self.d_seq.append(self._acquire(self.window_action))
        d = np.stack(self.d_seq)
        pos = np.asarray(self.pos_seq, int)

        # --- re-solve the blind problem on everything measured so far
        z0 = self.z_hat if self.warm_start else None
        w0 = self.w_hat if self.warm_start else None
        self.z_hat, self.w_hat, self.opt_res = solve_blind(
            d, pos, self.image_size, self.probe_size, z0=z0, w0=w0,
            penalty=self.penalty, method=self.solver, maxiter=self.solver_iters,
            periodic=self.periodic, fix_probe=not self.blind,
        )
        self.state = self._state_from(self.z_hat)

        self.reward = self._get_reward(d, pos)

        self._final_pos = pos            # kept for info() after the reset below

        if self.a_start > self.num_scans:
            self.a_start = self.init_start
            self.pos_seq, self.action_seq, self.d_seq = [], [], []
            self.done = True

        # return (np.array(self.state), self.reward, self.done,
        #         self.window_action, self.pos_seq, self.n)

        observation = self._get_obs()
        info = self._get_info()

        return observation, self.reward, self.done, False, info

    def render(self):
        pass

    ### Helper functions ###
    def candidate_windows(self, n, m, action_size, periodic=True, jitter=0, seed=0):
        """`action_size` candidate window corners on a (roughly) square grid.

        Action a  <->  S_{pos[a]}  <->  the slice z[r:r+m, c:c+m].
        A small jitter breaks the perfectly-periodic raster degeneracy that makes
        the *blind* problem ill-posed.
        """
        k = int(round(math.sqrt(action_size)))
        if k * k != action_size:
            raise ValueError("action_size must be a perfect square (grid of windows)")
        hi = n if periodic else n - m + 1
        g = np.unique(np.linspace(0, hi - 1, k).round().astype(int))
        pos = np.array([(r, c) for r in g for c in g], dtype=int)
        if jitter:
            pos += self.np_random.integers(-jitter, jitter + 1, pos.shape)
            pos = np.mod(pos, n) if periodic else np.clip(pos, 0, n - m)
        return pos

    def _make_objects(self, seed, n_images, n, image_type):
        if PhantomGenerator is not None:
            gen = PhantomGenerator(seed=seed, image_size=n)
            if image_type.value == ImageType.MIXED.value:
                P = gen.generate_mixed(n_samples=n_images)
            elif image_type.value == ImageType.TRIANGLE.value:
                P = gen.generate_triangle(n_samples=n_images)
            elif image_type.value == ImageType.TRIANGLE_THIRTY.value:
                P = gen.generate_triangle(n_samples=n_images, max_angle=30)
            elif image_type.value == ImageType.TRIANGLE_FIXED.value:
                P = gen.generate_triangle(n_samples=n_images, max_angle=0)
            elif image_type.value == ImageType.LINE.value:
                P = gen.generate_line(n_samples=n_images)
            elif image_type.value == ImageType.LINE_THIRTY.value:
                P = gen.generate_line(n_samples=n_images, max_angle=30)
            elif image_type.value == ImageType.LINE_FIXED.value:
                P = gen.generate_line(n_samples=n_images, max_angle=0)
            elif image_type.value == ImageType.SKIMAGE.value:
                P = gen.generate_skimage(n_samples=n_images)
            else:
                raise Exception("Unknown ImageType %s" % image_type)
            P = np.asarray(P, dtype=float)
        else:                                   # standalone fallback
            P = np.stack([self._smooth_field(n) for _ in range(n_images)])

        # phantom -> complex transmission function:  A e^{i phi}
        return (1.0 - 0.5 * P) * np.exp(1j * np.pi * (P - 0.5))

    def _smooth_field(self, n, bw=0.08):
        A = np.fft.fft2(self.np_random.normal(size=(n, n)))
        f = np.fft.fftfreq(n)
        g = np.exp(-(f[:, None] ** 2 + f[None, :] ** 2) / (2 * bw ** 2))
        o = np.real(np.fft.ifft2(A * g))
        return (o - o.min()) / (o.max() - o.min() + _EPS)

    def valid_action_mask(self):
        mask = np.ones(len(self.positions), dtype=bool)
        if not self.allow_repeats:
            mask[self.action_seq] = False
        return mask

    def _acquire(self, corner):
        """d_j: one phaseless diffraction pattern (magnitudes), with shot noise."""
        return measure(self.np_random, self.P_all[self.n], self.w_true, np.array([corner]),
                       self.periodic, photons=self.corruption_photons,
                       seed=int(self.np_random.integers(1 << 31)))[0]

    def _state_from(self, z):
        if self.state_mode == "amp":
            return np.abs(z).astype(np.float32)
        if self.state_mode == "ampphase":
            return np.stack([np.abs(z), np.angle(z) / np.pi]).astype(np.float32)
        return np.stack([z.real, z.imag]).astype(np.float32)   # "reim"

    # ---------------- reward ----------------
    def _fwd_error(self, pos, d):
        """-|| |F(w_hat .* S_j z_hat)| - d || / ||d||   (phaseless data misfit)."""
        err = la.norm(np.abs(forward(self.z_hat, self.w_hat, np.asarray(pos, int),
                                     self.periodic)) - d)
        return -err / max(la.norm(d), _EPS)

    def _rnd_positions(self, n_scans):
        idx = self.np_random.choice(len(self.positions),
                              size=min(n_scans, len(self.positions)), replace=False)
        return self.positions[idx]

    def _get_reward(self, d, pos):
        reward = 0.0
        rt, R = self.reward_type.value, RewardType

        if rt in (R.PNSR_E2E.value, R.PNSR_INCREMENTAL.value, R.PNSR_ADD.value):
            z_ref = self.P_all[self.n]
            current_reward = psnr(z_ref, align(self.z_hat, z_ref))
            if rt == R.PNSR_E2E and self.a_start > self.num_scans:
                reward = current_reward
            elif rt == R.PNSR_INCREMENTAL.value:
                reward = current_reward - self.previous_reward
                self.previous_reward = current_reward
            elif rt == R.PNSR_ADD.value:
                reward = current_reward / float(self.num_scans)

        elif rt in (R.FWD_E2E.value, R.FWD_INCREMENTAL.value):
            current_reward = self._fwd_error(pos, d)
            if rt == R.FWD_E2E.value and self.a_start > self.num_scans:
                reward = current_reward
            elif rt == R.FWD_INCREMENTAL.value:
                reward = current_reward - self.previous_reward
                self.previous_reward = current_reward

        elif rt in (R.FWD_RND_1.value, R.FWD_RND_10.value, R.FWD_RND_COMPLETE_10.value,
                    R.FWD_RND_COMPLETE_FULL.value, R.FWD_INC_RND_1.value, R.FWD_INC_RND_10.value):
            # "does the reconstruction predict windows we never scanned?"
            if rt in (R.FWD_RND_COMPLETE_10.value, R.FWD_RND_COMPLETE_FULL.value) \
                    and self.a_start < self.num_scans:
                return reward
            n_scans = 1 if rt in (R.FWD_RND_1.value, R.FWD_INC_RND_1.value) else 10
            rnd_pos = (self.positions if rt == R.FWD_RND_COMPLETE_FULL.value
                       else self._rnd_positions(n_scans))
            rnd_d = np.stack([self._acquire(p) for p in rnd_pos])
            current_reward = self._fwd_error(rnd_pos, rnd_d)

            if rt in (R.FWD_RND_1.value, R.FWD_RND_10.value):
                reward = current_reward / float(self.num_scans)
            elif rt in (R.FWD_INC_RND_1.value, R.FWD_INC_RND_10.value):
                reward = current_reward - self.previous_reward
                self.previous_reward = current_reward
            else:
                reward = current_reward

        elif rt == R.FWD_RND_1_REG.value:
            rnd_pos = self._rnd_positions(1)
            rnd_d = np.stack([self._acquire(p) for p in rnd_pos])
            z_ref = self.P_all[self.n]
            reward = (self._fwd_error(rnd_pos, rnd_d)
                      + self.pnsr_reg * psnr(z_ref, align(self.z_hat, z_ref)))
            reward /= float(self.num_scans)

        else:
            raise Exception("Unknown reward_type %s" % rt)

        return reward

    def info(self):
        pos = (np.asarray(self.pos_seq, int) if self.pos_seq
               else getattr(self, "_final_pos", np.zeros((0, 2), int)))
        cov = (coverage(self.image_size, self.probe_size, pos, self.periodic)
               if len(pos) else np.zeros((self.image_size,) * 2, int))
        z_ref = self.P_all[self.n]
        seen = cov > 0
        err = (la.norm((align(self.z_hat, z_ref) - z_ref)[seen])
               / max(la.norm(z_ref[seen]), _EPS)) if seen.any() else np.nan
        return {"n_scans": len(pos), "min_coverage": int(cov.min()),
                "covered_frac": float(seen.mean()), "rel_err_z_seen": float(err)}
