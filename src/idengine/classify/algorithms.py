"""Classification algorithms implemented with numpy.

Both models expose ``fit(X, y)``, ``predict_proba(X)`` and array-based state
for persistence. ``y`` holds integer class indices.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatMatrix = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]


def softmax(logits: FloatMatrix) -> FloatMatrix:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class Model(Protocol):
    name: str

    def fit(self, features: FloatMatrix, labels: IntVector, n_classes: int) -> list[float]: ...

    def predict_proba(self, features: FloatMatrix) -> FloatMatrix: ...

    def arrays(self) -> dict[str, npt.NDArray[np.float64]]: ...


class MultinomialNaiveBayes:
    """Multinomial NB with Lidstone smoothing over term counts."""

    name = "naive_bayes"

    def __init__(self, alpha: float = 0.5) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.class_log_prior: npt.NDArray[np.float64] = np.zeros(0)
        self.feature_log_prob: FloatMatrix = np.zeros((0, 0))

    def fit(self, features: FloatMatrix, labels: IntVector, n_classes: int) -> list[float]:
        n_features = features.shape[1]
        class_counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
        feature_counts = np.zeros((n_classes, n_features))
        np.add.at(feature_counts, labels, features)
        smoothed = feature_counts + self.alpha
        self.feature_log_prob = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        # Classes absent from training get a tiny prior instead of -inf.
        priors = np.maximum(class_counts, 1e-3) / max(class_counts.sum(), 1.0)
        self.class_log_prior = np.log(priors)
        return []

    def joint_log_likelihood(self, features: FloatMatrix) -> FloatMatrix:
        return features @ self.feature_log_prob.T + self.class_log_prior

    def predict_proba(self, features: FloatMatrix) -> FloatMatrix:
        return softmax(self.joint_log_likelihood(features))

    def arrays(self) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "class_log_prior": self.class_log_prior,
            "feature_log_prob": self.feature_log_prob,
        }

    @classmethod
    def from_arrays(
        cls, arrays: dict[str, npt.NDArray[np.float64]], alpha: float
    ) -> MultinomialNaiveBayes:
        model = cls(alpha)
        model.class_log_prior = arrays["class_log_prior"]
        model.feature_log_prob = arrays["feature_log_prob"]
        return model


class SoftmaxRegression:
    """Multinomial logistic regression trained with Adam and L2 regularisation."""

    name = "logistic"

    def __init__(
        self,
        *,
        l2: float = 1e-3,
        epochs: int = 300,
        learning_rate: float = 0.5,
        tol: float = 1e-6,
        seed: int = 0,
    ) -> None:
        self.l2 = l2
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tol = tol
        self.seed = seed
        self.weights: FloatMatrix = np.zeros((0, 0))
        self.bias: npt.NDArray[np.float64] = np.zeros(0)

    def _loss(self, probs: FloatMatrix, onehot: FloatMatrix) -> float:
        nll = -np.sum(onehot * np.log(np.clip(probs, 1e-12, 1.0))) / onehot.shape[0]
        return float(nll + 0.5 * self.l2 * np.sum(self.weights**2))

    def fit(self, features: FloatMatrix, labels: IntVector, n_classes: int) -> list[float]:
        n, d = features.shape
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.01, size=(d, n_classes))
        self.bias = np.zeros(n_classes)
        onehot = np.eye(n_classes)[labels]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_w = np.zeros_like(self.weights)
        v_w = np.zeros_like(self.weights)
        m_b = np.zeros_like(self.bias)
        v_b = np.zeros_like(self.bias)
        history: list[float] = []
        lr = self.learning_rate / 10.0  # Adam steps are scale-free; keep them modest
        for step in range(1, self.epochs + 1):
            probs = softmax(features @ self.weights + self.bias)
            error = (probs - onehot) / n
            grad_w = features.T @ error + self.l2 * self.weights
            grad_b = error.sum(axis=0)
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w**2
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b**2
            correction1 = 1 - beta1**step
            correction2 = 1 - beta2**step
            self.weights -= lr * (m_w / correction1) / (np.sqrt(v_w / correction2) + eps)
            self.bias -= lr * (m_b / correction1) / (np.sqrt(v_b / correction2) + eps)
            loss = self._loss(probs, onehot)
            history.append(loss)
            if step > 10 and abs(history[-2] - loss) < self.tol:
                break
        return history

    def predict_proba(self, features: FloatMatrix) -> FloatMatrix:
        return softmax(features @ self.weights + self.bias)

    def arrays(self) -> dict[str, npt.NDArray[np.float64]]:
        return {"weights": self.weights, "bias": self.bias}

    @classmethod
    def from_arrays(cls, arrays: dict[str, npt.NDArray[np.float64]]) -> SoftmaxRegression:
        model = cls()
        model.weights = arrays["weights"]
        model.bias = arrays["bias"]
        return model
