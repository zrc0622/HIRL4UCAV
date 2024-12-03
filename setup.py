from setuptools import setup, find_packages

setup(
    name='hirl',
    version='0.1',
    packages=find_packages(),  # 自动发现所有子包
    install_requires=[
        "torch>=1.13.0",
        "matplotlib>=3.7.5",
        "pandas>=2.0.3",
        "gym>=0.26.2",
        "pyyaml>=6.0.2",
        "tensorboard==2.13.0"
    ],
)
