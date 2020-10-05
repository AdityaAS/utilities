# utilities

This repository contains utilities typically used in the development of deep learning algorithms (in PyTorch)

Use the repository to avoid writing boilerplate code for data processing/logging/metrics and visualization (at the moment)

The project currently has limited functionality but will increase over time

```
.
├── __init__.py
├── install.sh
├── LICENSE
├── README.md
├── setup.py
└── utilities
    ├── factory.py
    ├── image.py
    ├── __init__.py
    ├── io.py
    ├── logger.py
    ├── metrics.py
    ├── random.py
    ├── utils.py
    └── viz.py
```

## Setup & Installation
* The repository is pip installable and can be imported as `import utilities` once installed
* Run `bash install.sh` to install the package

## Examples
1. To set seeds for your run (for reproducibility)

```
from utilities.random import seed_everything

seed_everything(seed=4321)
```

2. To convert class labels to one hot encoding

```
from utilities.utils import make_one_hot
one_hot = make_one_hot(position=2, length=5) # Output will be the tensor [0 0 1 0 0]
```

---
## Contributors
- [Aditya Sarma](https://adityaas.github.io/)
