import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vip_ivp",
    version="0.1.0",
    author="Nathan Gripon",
    author_email="n.gripon@gmail.com",
    description="Solve ODEs without having to build the system of equations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ngripon/vip-ivp",
    project_urls={
        "Bug Tracker": "https://github.com/ngripon/vip-ivp/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=['numpy', 'scipy', 'matplotlib']
)
