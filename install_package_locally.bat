call .venv\Scripts\activate.bat

pip uninstall vip_ivp -y
python -m build
pip install dist/vip_ivp-0.2.0.tar.gz