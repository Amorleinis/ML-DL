from setuptools import setup, find_packages

setup(
    name='ML-DL',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'joblib',
        'pandas',
        'scikit-learn',
        'tensorflow',
    ],
    entry_points={
        'console_scripts': [
            # Add console script entry points here if needed
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for machine learning and deep learning models',
    url='https://github.com/Amorleinis/ML-DL',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
