call ..\.venv\Scripts\activate.bat

sphinx-build -M markdown ./Sphinx-docs ./Sphinx-docs/build

xcopy "Sphinx-docs\build\markdown\vip_ivp.md" "website\docs\" /Y