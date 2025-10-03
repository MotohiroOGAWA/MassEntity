English | [日本語](README.ja.md)

# MassEntity

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange)

---

MassEntity is a Python library for handling mass spectrometry data.  
You can install it directly from GitHub with pip, or integrate it into your project as a Git submodule for development.

---

## 🔗 Add as a Git Submodule

To integrate MassEntity into an existing project, run the following commands at the root of your project:

```bash
# At the root directory of your project
git submodule add https://github.com/MotohiroOGAWA/MassEntity.git ./MassEntity
git commit -m "Add MassEntity as submodule"
```

## ⚙️ Install from GitHub (Recommended for Users)
You can install **MassEntity** directly from the GitHub repository:
```bash
pip install git+https://github.com/MotohiroOGAWA/MassEntity.git
```

## 🧪 Running Tests
You can run tests to verify that MassEntity is working properly.

### Using Python
```bash
python -m run_tests
```

### Using the shell script
```bash
./run_tests.sh
```