[English](README.md) | 日本語

# MassEntity

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange)

---

MassEntity は質量分析データを扱うための Python ライブラリです。  
Git サブモジュールとして組み込むことや、開発モードでのインストールが可能です。

---

## 🔗 Git サブモジュールとして導入

既存のプロジェクトに MassEntity を組み込みたい場合は、プロジェクトのルートディレクトリで以下を実行してください。

```bash
# プロジェクトのルートディレクトリで
git submodule add https://github.com/your-username/MassEntity.git cores/MassEntity
git commit -m "Add MassEntity as submodule"
```

## 🔄 サブモジュールの更新方法
```bash
cd cores/MassEntity
git checkout main
git pull origin main
cd ../..
git add cores/MassEntity
git commit -m "Update MassEntity submodule"
```

## ⚙️ インストール（開発モード）
MassEntity を Python 環境にインストールするには以下を実行します。
```bash
cd cores/MassEntity
pip install -e .
```
-e オプション（editable mode）を付けることで、ソースコードを編集した変更が即時に反映されます。


## 🧪 テストの実行
MassEntity が正しく動作するか確認するためにテストを実行できます。

### Python から実行する場合

python run_tests.py

### シェルスクリプトを使う場合

./run_tests.sh