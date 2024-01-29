import setuptools

with open("README.md") as fh:
    long_description = fh.read()

install_requires = [
    "numpy >= 1.19.4",
    "scipy >= 1.5.4",
    "scikit-learn",
    "scikit-image",
    "fbpca >= 1.0",
    "joblib",
    "h5py",
    "astropy",
    "hdfdict",
    "matplotlib",
]

# extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})

packages = [
    "hannibal_specter",
    "hannibal_specter.decomposition",
    "hannibal_specter.data",
    "hannibal_specter.fake",
    "hannibal_specter.file",
    "hannibal_specter.gridsearch",
    "hannibal_specter.template",
    "hannibal_specter.preprocess",
    "hannibal_specter.plot",
    "hannibal_specter.quality",
    "hannibal_specter.varia",
    "hannibal_specter.math",
]

tests = [p + ".tests" for p in packages]

setuptools.setup(
    name="hannibal-specter",
    version="0.0.1",
    author="R.A.R. MOENS",
    author_email="rogermoens@delfius.nl",
    description="A small package for robust matrix decomposition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RogerMoens/hannibal-specter",
    packages=packages + tests,
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
