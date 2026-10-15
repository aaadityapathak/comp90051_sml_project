"""The project's fixed-budget binary classifier using the official TabM package.

Train each of the 16 members with BCE, average losses within a record, and
then apply the class/view weight supplied by the shared input workflow. At
inference, average member probabilities, not logits. Architecture reference:
https://github.com/yandex-research/tabm (pinned in algorithms/tabm/requirements.txt).
"""
from importlib.metadata import version
from pathlib import Path
import os
import random
import time

# Required by CUDA's deterministic matrix multiplication. Set before CUDA use.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch.nn import functional as F
from tabm import TabM


def memberwise_weighted_loss(logits, target, normalized_weight):
    """BCE averaged over members, then weighted and averaged over records.

    ``normalized_weight`` is weight / mean(weight) for the ENTIRE training
    dataset, not for this minibatch. This makes each minibatch gradient an
    unbiased estimate of the dataset's weighted mean loss under shuffling.
    ``logits`` has shape (records, members, 1).
    """
    members = logits.squeeze(-1)
    losses = F.binary_cross_entropy_with_logits(
        members, target[:, None].expand_as(members), reduction='none')
    return (losses.mean(dim=1) * normalized_weight).mean()


def _device(requested):
    resolved = ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested
    device = torch.device(resolved)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices are auto, cpu, cuda, or cuda:<index>.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; use device="cpu" or "auto".')
    return device


class TabMClassifier:
    """A reproducible TabM fit with a fixed epoch budget and no early stopping."""

    def __init__(self, n_blocks=2, epochs=100, seed=20261008, device='auto',
                 batch_size=256, verbose=True):
        for name, value in [('n_blocks', n_blocks), ('epochs', epochs), ('batch_size', batch_size)]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32).')
        self.n_blocks, self.epochs, self.seed = int(n_blocks), int(epochs), int(seed)
        self.device, self.batch_size, self.verbose = str(device), int(batch_size), bool(verbose)

    def _architecture(self, n_features):
        return dict(arch_type='tabm', n_num_features=n_features, d_out=1,
                    n_blocks=self.n_blocks, d_block=128, k=16, dropout=0.1,
                    num_embeddings=None, cat_cardinalities=None)

    def _make_model(self):
        return TabM.make(**self._architecture(self.n_features_in_)).to(self.device_)

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32, order='C')
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 1:
            raise ValueError('Training targets must be a one-dimensional binary array.')
        weight = np.ones(len(y), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if (X.ndim != 2 or not len(X) or not X.shape[1] or y.shape != (len(X),)
                or weight.shape != y.shape or not np.isfinite(X).all()
                or not np.isfinite(weight).all() or (weight <= 0).any()
                or set(np.unique(y)) != {0, 1}):
            raise ValueError('Finite 2D predictors, both binary classes, aligned positive weights required.')
        weight_mean = float(weight.mean())
        normalized_weight = np.asarray(weight / weight_mean, dtype=np.float32)
        if not np.isfinite(weight_mean) or not np.isfinite(normalized_weight).all() or (normalized_weight <= 0).any():
            raise ValueError('Weights must have a finite positive mean and be representable in float32.')

        self.__dict__.pop('audit_', None)
        started = time.perf_counter()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device_ = _device(self.device)
        self.n_features_in_ = X.shape[1]
        self.model_ = self._make_model()
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=3e-4, weight_decay=1e-4)

        # Cache full training tensors on GPU when they fit comfortably. Larger
        # datasets remain in host RAM and only their minibatches are transferred.
        data_device = torch.device('cpu')
        if self.device_.type == 'cuda':
            free_bytes, _ = torch.cuda.mem_get_info(self.device_)
            required_bytes = X.nbytes + y.nbytes + normalized_weight.nbytes + 8 * len(X)
            if required_bytes < min(free_bytes * 0.4, 1024**3):
                data_device = self.device_
        x_tensor = torch.as_tensor(X, device=data_device)
        y_tensor = torch.as_tensor(y, device=data_device)
        w_tensor = torch.as_tensor(normalized_weight, device=data_device)
        generator = torch.Generator(device=data_device).manual_seed(self.seed)
        self.history_ = []
        self.model_.train()
        for epoch in range(1, self.epochs + 1):
            epoch_started = time.perf_counter()
            order = torch.randperm(len(X), generator=generator, device=data_device)
            epoch_loss = torch.zeros((), device=self.device_)
            for indices in order.split(self.batch_size):
                xb = x_tensor[indices].to(self.device_)
                yb = y_tensor[indices].to(self.device_)
                wb = w_tensor[indices].to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(xb, x_cat=None)
                loss = memberwise_weighted_loss(logits, yb, wb)
                loss.backward()
                optimizer.step()
                # Synchronize only once per epoch, not once per minibatch.
                epoch_loss += loss.detach() * len(indices)
            weighted_bce = float((epoch_loss / len(X)).cpu())
            if not np.isfinite(weighted_bce):
                raise RuntimeError(f'Non-finite TabM training loss at epoch {epoch}.')
            self.history_.append({'epoch': epoch, 'weighted_bce': weighted_bce,
                                  'seconds': time.perf_counter() - epoch_started})
            if self.verbose and (epoch == 1 or epoch % 10 == 0 or epoch == self.epochs):
                print(f'TabM blocks={self.n_blocks} seed={self.seed} epoch={epoch}/{self.epochs} '
                      f'weighted_bce={weighted_bce:.6f}', flush=True)
        self.model_.eval()
        self.audit_ = {
            'n_blocks': self.n_blocks, 'epochs': self.epochs, 'epochs_completed': len(self.history_),
            'seed': self.seed, 'batch_size': self.batch_size, 'n_train_views': len(X),
            'n_features': self.n_features_in_, 'weight_sum': float(weight.sum()),
            'normalized_weight_mean': float(normalized_weight.mean(dtype=np.float64)),
            'weight_normalization': 'training_dataset_mean',
            'objective': 'memberwise BCE mean, then sample-weighted record mean',
            'prediction': 'mean member sigmoid probabilities', 'early_stopping': False,
            'mixed_precision': False, 'deterministic_algorithms': True,
            'device': str(self.device_), 'data_device': str(data_device),
            'device_name': torch.cuda.get_device_name(self.device_) if self.device_.type == 'cuda' else 'CPU',
            'train_seconds': time.perf_counter() - started,
            'final_training_weighted_bce': self.history_[-1]['weighted_bce'],
            'architecture': self._architecture(self.n_features_in_),
            'optimizer': {'name': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4},
            'versions': {'torch': str(torch.__version__), 'tabm': version('tabm'),
                         'numpy': str(np.__version__), 'cuda': torch.version.cuda},
        }
        return self

    def predict_proba(self, X):
        if not hasattr(self, 'model_'):
            raise RuntimeError('Fit or load a TabMClassifier before predicting.')
        X = np.asarray(X, dtype=np.float32, order='C')
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError('Prediction matrix must match finite fitted predictors.')
        if not len(X):
            return np.empty(0, dtype=np.float32)
        self.model_.eval()
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(X), self.batch_size):
                batch = torch.as_tensor(X[start:start + self.batch_size], device=self.device_)
                logits = self.model_(batch, x_cat=None)
                predictions.append(logits.sigmoid().mean(dim=1).squeeze(-1).cpu().numpy())
        result = np.concatenate(predictions)
        if not np.isfinite(result).all():
            raise RuntimeError('TabM produced non-finite probabilities.')
        return result

    def save(self, path):
        """Atomically save portable CPU weights, configuration, and training audit."""
        if not hasattr(self, 'audit_'):
            raise RuntimeError('Only a completed fit can be saved.')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'format_version': 1,
            'parameters': {'n_blocks': self.n_blocks, 'epochs': self.epochs, 'seed': self.seed,
                           'batch_size': self.batch_size, 'verbose': self.verbose},
            'n_features': self.n_features_in_, 'history': self.history_, 'audit': self.audit_,
            'state_dict': {name: tensor.detach().cpu() for name, tensor in self.model_.state_dict().items()},
        }
        temporary = path.with_name(path.name + '.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, device='auto'):
        """Restore an inference checkpoint; it does not resume optimizer training."""
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload['format_version'] != 1:
            raise ValueError('Unsupported TabM checkpoint format.')
        fitted = cls(**payload['parameters'], device=device)
        fitted.device_ = _device(device)
        fitted.n_features_in_ = payload['n_features']
        fitted.model_ = fitted._make_model()
        fitted.model_.load_state_dict(payload['state_dict'])
        fitted.model_.eval()
        fitted.history_, fitted.audit_ = payload['history'], payload['audit']
        return fitted
