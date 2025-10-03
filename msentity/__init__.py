# msentity/__init__.py
import importlib

core = importlib.import_module(".core", __package__)
io = importlib.import_module(".io", __package__)
utils = importlib.import_module(".utils", __package__)
