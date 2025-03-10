call .venv\Scripts\activate.bat

pip uninstall vip_ivp -y
python -m build
pip install dist/vip_ivp-0.1.1.tar.gz