# Label Encoders

A scikit-learn compatible label encoder with random and manual encoding strategies.

## Overview

This package provides two label encoder transformers that extend scikit-learn's `BaseEstimator` and `TransformerMixin`:

- **LabelEncoderRandom** — Encodes target labels with randomly shuffled integer values, avoiding the ordinal relationship implied by sequential encoding.
- **LabelEncoderManual** — Encodes target labels using a user-supplied mapping dictionary.

## Installation

### From PyPI

```bash
pip install label_encoder_random
```

### From Source (Development)

```bash
git clone https://github.com/ptrajdos/LabelEncoderRandom.git
cd LabelEncoderRandom
make venv        # Create a virtual environment
make pypackages  # Install the package in editable mode with dev dependencies
```

Or manually:

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install wheel setuptools
pip install -e . -r requirements_dev.txt
```

### Requirements

- numpy >= 1.22.4
- scikit-learn >= 1.2.2
- joblib >= 1.2.0

## Usage

### LabelEncoderRandom

```python
from label_encoder_random.transformers.label_encoder_random import LabelEncoderRandom

le = LabelEncoderRandom(offset=0, randomize=True)
le.fit(["cat", "dog", "cat", "bird"])

encoded = le.transform(["cat", "dog"])
original = le.inverse_transform(encoded)
```

**Parameters:**

- `offset` (int, default=0) — Value added to every encoded label.
- `randomize` (bool, default=True) — If `True`, integer codes are randomly shuffled; if `False`, labels are encoded sequentially.
- `disable_check` (bool, default=False) — If `True`, `inverse_transform` skips validation of unseen labels.

### LabelEncoderManual

```python
from label_encoder_random.transformers.label_encoder_manual import LabelEncoderManual

le = LabelEncoderManual({"cat": 10, "dog": 20, "bird": 30})
le.fit(["cat", "dog", "cat", "bird"])

encoded = le.transform(["cat", "dog"])  # array([10, 20])
original = le.inverse_transform([10, 20])  # array(['cat', 'dog'])
```

**Parameters:**

- `mapping` (dict) — A dictionary defining the one-to-one mapping between original labels (keys) and encoded integer values (values).

## Development

### Dev Requirements

Install development dependencies (via Make):

```bash
make pypackages
```

Or manually:

```bash
pip install -r requirements_dev.txt
```

### Running Tests

```bash
make test             # Run tests with coverage
make test_parallel    # Run tests in parallel with coverage
```

Or manually:

```bash
python -m pytest tests/
```

### Generate Documentation

```bash
make docs
```

### Generate UML Diagrams

```bash
make uml
```

Generates UML class diagrams (SVG) via `pyreverse` into the `uml/` directory.
Requires [Graphviz](https://graphviz.org/) to be installed on the system (`dot` command must be available on `PATH`).

### Linting / Multi-environment Testing

```bash
make tox_check
```

### Cleanup

```bash
make clean
```

## License

See [LICENSE](LICENSE) for details.

## Author

Pawel Trajdos — [pawel.trajdos@pwr.edu.pl](mailto:pawel.trajdos@pwr.edu.pl)

**Repository:** [https://github.com/ptrajdos/LabelEncoderRandom](https://github.com/ptrajdos/LabelEncoderRandom)
