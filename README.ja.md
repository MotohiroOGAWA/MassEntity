[English](README.md) | 日本語

# MassEntity

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange)

---

MassEntity は質量分析データを扱うための Python ライブラリです。  
pip を使って GitHub から直接インストールすることも、開発用にプロジェクトへ Git サブモジュールとして統合することもできます。

---

## 🔗 Git サブモジュールとして導入

既存のプロジェクトに MassEntity を組み込みたい場合は、プロジェクトのルートディレクトリで以下を実行してください。

```bash
# プロジェクトのルートディレクトリで
git submodule add https://github.com/MotohiroOGAWA/MassEntity.git ./MassEntity
git commit -m "Add MassEntity as submodule"
```

## ⚙️ GitHub からインストール（ユーザー向け推奨）
MassEntity は GitHub リポジトリから直接インストールできます。
```bash
pip install git+https://github.com/MotohiroOGAWA/MassEntity.git
```


## 🧪 テストの実行
MassEntity が正しく動作するか確認するために、テストを実行できます。

### Python から実行する場合
```
python -m run_tests
```

### シェルスクリプトを使う場合
```bash
./run_tests.sh
```