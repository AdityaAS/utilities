pip uninstall utilities
python setup.py sdist bdist_wheel
pip install dist/utilities-0.0.1-py3-none-any.whl
rm -rf build dist utilities.egg-info
