"""Six-DoF Semantic-Adaptive Particle Filter (SAPF).

The state is [x, y, z, roll, pitch, yaw].  The filter uses a diagonal process
noise model Q(C_t), where C_t is a discrete terrain label produced by the
AHPFF front end.  Stairs increase vertical and pitch uncertainty; obstacles
increase horizontal and yaw uncertainty.  The boost arrays are
standard-deviation equivalents and their variances are added:
Q = diag(base_std^2 + boost^2).  Systematic resampling is triggered when the
effective sample size falls below a configurable fraction of the particle
count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STATE_DIM = 6
POSITION = slice(0, 3)
ORIENTATION = slice(3, 6)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angles(values: np.ndarray) -> np.ndarray:
    wrapped = np.array(values, copy=True)
    wrapped[..., 3:6] = wrap_angle(wrapped[..., 3:6])
    return wrapped


def circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    sine = float(np.sum(weights * np.sin(angles)))
    cosine = float(np.sum(weights * np.cos(angles)))
    return float(np.arctan2(sine, cosine))


@dataclass
class SAPFConfig:
    num_particles: int = 1000
    process_std_base: np.ndarray = field(
        default_factory=lambda: np.array([0.002, 0.002, 0.001, 0.001, 0.001, 0.001])
    )
    process_std_boost: dict[int, np.ndarray] = field(
        default_factory=lambda: {
            0: np.zeros(STATE_DIM),
            1: np.array(
                [0.0022360679775, 0.0022360679775, 0.0059160797831, 0.0038729833462, 0.0109544511501, 0.0017320508076]
            ),
            2: np.array(
                [0.0118321595662, 0.0118321595662, 0.0017320508076, 0.0059160797831, 0.0017320508076, 0.0109544511501]
            ),
            3: np.array(
                [0.005, 0.005, 0.002, 0.002, 0.002, 0.005]
            ),
        }
    )
    resample_threshold: float = 0.5
    roughening_std: float = 0.0
    use_semantics: bool = True


class SemanticAdaptiveParticleFilter:
    def __init__(self, config: SAPFConfig, rng: np.random.RandomState):
        self.config = config
        self.rng = rng
        self.particles = np.zeros((config.num_particles, STATE_DIM), dtype=np.float64)
        self.weights = np.full(config.num_particles, 1.0 / config.num_particles)
        self.initialized = False
        self.resample_count = 0
        self.ess_history: list[float] = []

    def initialize(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.particles = mean[None, :] + self.rng.normal(
            0.0, 1.0, size=(self.config.num_particles, STATE_DIM)
        ) * std[None, :]
        self.particles = wrap_angles(self.particles)
        self.weights = np.full(self.config.num_particles, 1.0 / self.config.num_particles)
        self.initialized = True

    def process_std(self, semantic_label: int) -> np.ndarray:
        base = self.config.process_std_base
        if not self.config.use_semantics:
            return base
        boost = self.config.process_std_boost.get(int(semantic_label), np.zeros(STATE_DIM))
        return np.sqrt(base**2 + boost**2)

    def maximum_process_std(self) -> np.ndarray:
        maximum_boost = np.zeros(STATE_DIM)
        for boost in self.config.process_std_boost.values():
            maximum_boost = np.maximum(maximum_boost, np.asarray(boost, dtype=np.float64))
        return np.sqrt(self.config.process_std_base**2 + maximum_boost**2)

    def predict(self, control_delta: np.ndarray, semantic_label: int) -> None:
        if not self.initialized:
            raise RuntimeError("Call initialize() before predict().")
        q_std = self.process_std(semantic_label)
        self.particles = (
            self.particles
            + control_delta[None, :]
            + self.rng.normal(0.0, 1.0, size=(self.config.num_particles, STATE_DIM)) * q_std[None, :]
        )
        self.particles = wrap_angles(self.particles)

    def update(self, observation: np.ndarray, observation_std: np.ndarray) -> None:
        difference = observation[None, :] - self.particles
        difference = wrap_angles(difference)
        log_likelihood = -0.5 * np.sum((difference / observation_std[None, :]) ** 2, axis=1)
        log_likelihood -= np.max(log_likelihood)
        updated = self.weights * np.exp(log_likelihood)
        total = float(np.sum(updated))
        if not np.isfinite(total) or total <= 1e-300:
            self.weights = np.full(self.config.num_particles, 1.0 / self.config.num_particles)
        else:
            self.weights = updated / total

    def estimate(self) -> np.ndarray:
        estimate = np.zeros(STATE_DIM, dtype=np.float64)
        estimate[POSITION] = np.sum(self.weights[:, None] * self.particles[:, POSITION], axis=0)
        for index in range(3, 6):
            estimate[index] = circular_mean(self.particles[:, index], self.weights)
        return estimate

    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights**2))

    def systematic_resample(self) -> None:
        count = self.config.num_particles
        positions = (self.rng.random() + np.arange(count)) / count
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, positions)
        self.particles = self.particles[indices]
        self.weights = np.full(count, 1.0 / count)
        if self.config.roughening_std > 0.0:
            q_std = self.config.process_std_base * self.config.roughening_std
            self.particles = self.particles + self.rng.normal(
                0.0, 1.0, size=(count, STATE_DIM)
            ) * q_std[None, :]
            self.particles = wrap_angles(self.particles)
        self.resample_count += 1

    def step(
        self,
        control_delta: np.ndarray,
        observation: np.ndarray,
        observation_std: np.ndarray,
        semantic_label: int,
    ) -> np.ndarray:
        self.predict(control_delta, semantic_label)
        self.update(observation, observation_std)
        estimate = self.estimate()
        self.ess_history.append(self.effective_sample_size())
        if self.effective_sample_size() < self.config.resample_threshold * self.config.num_particles:
            self.systematic_resample()
        return estimate


class SixDoFExtendedKalmanFilter:
    """Additive 6-DoF Kalman filter used as a non-particle baseline.

    The motion and observation models are linear in the state, so this is the
    exact Kalman filter for the additive model used by the SAPF simulation.
    Its process covariance is fixed and does not use the terrain label.
    """

    def __init__(self, process_std: np.ndarray, rng: np.random.RandomState | None = None):
        self.process_std = np.asarray(process_std, dtype=np.float64)
        self.rng = rng or np.random.RandomState(0)
        self.mean = np.zeros(STATE_DIM, dtype=np.float64)
        self.covariance = np.eye(STATE_DIM, dtype=np.float64)
        self.initialized = False

    def initialize(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float64).copy()
        self.covariance = np.diag(np.asarray(std, dtype=np.float64) ** 2)
        self.initialized = True

    def predict(self, control_delta: np.ndarray, semantic_label: int = 0) -> None:
        if not self.initialized:
            raise RuntimeError("Call initialize() before predict().")
        self.mean = wrap_angles(self.mean + np.asarray(control_delta, dtype=np.float64))
        self.covariance = self.covariance + np.diag(self.process_std**2)

    def update(self, observation: np.ndarray, observation_std: np.ndarray) -> None:
        observation = np.asarray(observation, dtype=np.float64)
        observation_std = np.asarray(observation_std, dtype=np.float64)
        innovation = wrap_angle(observation - self.mean)
        innovation_covariance = self.covariance + np.diag(observation_std**2)
        gain = self.covariance @ np.linalg.inv(innovation_covariance)
        self.mean = wrap_angles(self.mean + gain @ innovation)
        identity = np.eye(STATE_DIM, dtype=np.float64)
        self.covariance = (identity - gain) @ self.covariance

    def step(
        self,
        control_delta: np.ndarray,
        observation: np.ndarray,
        observation_std: np.ndarray,
        semantic_label: int = 0,
    ) -> np.ndarray:
        self.predict(control_delta, semantic_label)
        self.update(observation, observation_std)
        return self.estimate()

    def estimate(self) -> np.ndarray:
        return self.mean.copy()
