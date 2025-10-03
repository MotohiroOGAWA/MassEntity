English | [日本語](README.ja.md)

# MassEntity

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange)

---

MassEntity is a Python library for handling mass spectrometry data.  
It can be integrated as a Git submodule or installed in editable mode.

---

## 🔗 Add as a Git Submodule

To integrate MassEntity into an existing project, run the following commands at the root of your project:

```bash
# At the root directory of your project
git submodule add https://github.com/MotohiroOGAWA/MassEntity.git cores/MassEntity
git commit -m "Add MassEntity as submodule"
```

## 🔄 Updating Submodules
```bash
cd cores/MassEntity
git checkout main
git pull origin main
cd ../..
git add cores/MassEntity
git commit -m "Update MassEntity submodule"
```

### Install from GitHub (Recommended for Users)
You can install **MassEntity** directly from the GitHub repository:
```bash
pip install git+https://github.com/MotohiroOGAWA/MassEntity.git
```

## ⚙️ Installation (Editable Mode)
To install MassEntity into your Python environment for development, run:
```bash
cd cores/MassEntity
pip install -e .
```
The -e option (editable mode) ensures that any modifications to the source code are immediately reflected in your environment.


## 🧪 Running Tests
You can run tests to verify that MassEntity is working properly.

### Using Python

python -m MassEntity.run_tests

### Using the shell script

./MassEntity/run_tests.sh