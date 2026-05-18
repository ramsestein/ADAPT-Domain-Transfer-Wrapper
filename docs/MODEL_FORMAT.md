# ADAPT — Model format guide

ADAPT can load a frozen model from any of five backends.  The model is
**never** retrained — it must already be trained and serialised before
running ADAPT.

---

## Supported backends at a glance

| File extension(s) | Backend | `type:` in config | Notes |
|---|---|---|---|
| `.json` `.ubj` `.bin` | XGBoost | `xgboost` | Recommended; native NaN handling |
| `.joblib` `.pkl` | sklearn (or any picklable object) | `sklearn` | Needs `.predict_proba()` |
| `.h5` `.keras` | Keras / TensorFlow | `keras` | NaNs imputed to 0 before inference |
| `.pt` `.pth` | PyTorch | `torch` | Must be a full model, not a `state_dict` |
| `<any>` + `custom_loader:` | BYOM | `byom` | You provide a Python loader |

The `type:` key is optional — ADAPT auto-detects the backend from the file
extension.  Specify it explicitly to override or when using BYOM.

---

## XGBoost

### Saving correctly

```python
import xgboost as xgb
booster = xgb.train(params, dtrain)
booster.save_model("inputs/models/my_model.json")   # preferred
# booster.save_model("inputs/models/my_model.ubj")  # binary alternative
```

For sklearn-API wrappers:

```python
from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X_train, y_train)
clf.save_model("inputs/models/my_model.json")
```

### Config

```yaml
model:
  path: inputs/models/my_model.json
  type: xgboost   # optional
```

### NaN handling

XGBoost has native NaN support — missing values are routed through learned
default directions in each split.  ADAPT passes the feature matrix as-is
without imputation.

### Caveats

- `.pkl` XGBoost models from old versions (< 1.6) may not restore correctly.
  Use `.json` instead.
- Multi-output models are not supported; the wrapper expects a single positive-
  class probability output.

---

## sklearn (and any object with `.predict_proba`)

### Saving correctly

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
joblib.dump(clf, "inputs/models/my_model.joblib")
```

Any Python object that implements `.predict_proba(X) -> (n, 2)` works —
pipeline, calibrated classifier, custom wrapper.

### Config

```yaml
model:
  path: inputs/models/my_model.joblib
  type: sklearn   # optional; also detected from .pkl extension
```

### NaN handling

sklearn models generally do not handle NaNs.  Impute upstream or use a
pipeline that includes an imputer step before loading.

---

## Keras / TensorFlow

### Saving correctly

```python
model.save("inputs/models/my_model.keras")   # recommended (SavedModel v3)
# model.save("inputs/models/my_model.h5")    # legacy HDF5 also accepted
```

### Config

```yaml
model:
  path: inputs/models/my_model.keras
  type: keras   # optional; detected from .h5 / .keras extension
```

### NaN handling

Keras models error on NaN inputs.  ADAPT automatically imputes NaN values
to **0** before calling the model.  This is a fallback — if your model is
sensitive to imputation, handle missingness upstream.

### Caveats

- The model must output a single value per sample (sigmoid activation) or a
  two-column array where column index 1 is the positive-class probability.
- Requires `tensorflow >= 2.12` at runtime.

---

## PyTorch

### Saving correctly

```python
import torch
torch.save(model, "inputs/models/my_model.pt")   # full model, not state_dict
```

Do **not** save only the state dict (`torch.save(model.state_dict(), ...)`)
because ADAPT cannot reconstruct the architecture without the class definition.

### Config

```yaml
model:
  path: inputs/models/my_model.pt
  type: torch   # optional; detected from .pt / .pth extension
```

### NaN handling

NaN values are imputed to **0** before inference (same as Keras).

### Caveats

- The model must expose a `.forward(x)` method that returns logits or
  probabilities for binary classification.
- ADAPT calls `torch.sigmoid` on the output if the range is outside [0, 1].
- Requires `torch >= 2.0` at runtime.

---

## BYOM — Bring Your Own Model

Use BYOM when your model does not fit any standard backend (e.g., ensemble
of heterogeneous models, R model loaded via `rpy2`, remote API).

### Loader skeleton

Create a Python file (e.g., `inputs/models/my_loader.py`) with exactly this
function signature:

```python
# inputs/models/my_loader.py

def load_model(path: str):
    """
    Load and return any model object from `path`.

    The returned object MUST expose:
        .predict_proba(X: np.ndarray | pd.DataFrame) -> np.ndarray
    where the output is either shape (n,) (positive-class probability)
    or shape (n, 2) ([:,1] is the positive-class probability).
    """
    import joblib
    model = joblib.load(path)
    # ... any custom logic ...
    return model
```

### Config

```yaml
model:
  path: inputs/models/my_model.pkl
  type: byom
  custom_loader: inputs/models/my_loader.py
```

### Minimal runnable example

```python
# inputs/models/ensemble_loader.py

def load_model(path: str):
    import joblib, numpy as np

    class EnsembleWrapper:
        def __init__(self, models):
            self.models = models
        def predict_proba(self, X):
            preds = np.stack([m.predict_proba(X)[:, 1] for m in self.models])
            return preds.mean(axis=0)

    return EnsembleWrapper(joblib.load(path))
```

### NaN handling

BYOM: ADAPT does **not** apply any automatic NaN imputation.  Your `load_model`
return object must handle NaNs however the original model expects.

---

## Choosing the right backend

```
Model was trained with XGBoost?          → XGBoost (.json preferred)
Model is a sklearn Pipeline or class?    → sklearn (.joblib preferred)
Model is a Keras/TF network?             → Keras (.keras preferred)
Model is a PyTorch Module?               → PyTorch (.pt)
Anything else?                           → BYOM
```

When in doubt, wrap the model in a sklearn-compatible class with
`.predict_proba()` and serialise with `joblib.dump` — that always works.

---

*Back to [ARCHITECTURE.md](ARCHITECTURE.md) · [USAGE.md](USAGE.md)*
