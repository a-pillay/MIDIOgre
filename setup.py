# Derived from: https://github.com/iver56/audiomentations/blob/main/setup.py

import os
from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

# Use environment variable to determine package name
# This allows different names for TestPyPI vs PyPI
package_name = os.environ.get('PACKAGE_NAME', 'midiogre')

setup(
    name=package_name,
    packages=find_packages(exclude=["demo", "tests"]),
    version='0.1.0',
    license='MIT',
    description='The On-the-fly MIDI Data Augmentation Library!',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ashwin Pillay',
    url='https://github.com/a-pillay/MIDIOgre',
    keywords=['MIDI', 'Audio', 'Machine Learning', 'Data Augmentation', 'Deep Learning'],
    install_requires=[
        "numpy>=1.21.6",
        "pretty-midi @ git+https://github.com/craffel/pretty-midi",
        "torch>=1.13.1",
    ],
    extras_require={
        "extras": [
            "matplotlib>=3.5.3",
        ]
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        # "Documentation": "TBA",  # TODO
        # "Changelog": "TBA",  # TODO
        "Issue Tracker": "https://github.com/a-pillay/MIDIOgre/issues",
    },
)
