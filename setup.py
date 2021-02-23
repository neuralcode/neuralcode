from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    name='neuralcode',
    version='0.0.1',
    description='A library for machine learning on source code',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_context_type='text/markdown',
    author='Ben Trevett',
    author_email='bentrevett@gmail.com',
    url='https://github.com/neuralcode/neuralcode',
    packages=find_packages(exclude='tests'),
    license='MIT',
    install_requires=required,
    include_package_data=True,
    python_requires='>=3.6',
)
