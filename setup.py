from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='OpenWorldRecognition',
    version='0.0.1',
    description='Open world recognition package',
    long_description=readme,
    author='David ter Kuile',
    author_email='doterkuile@gmail.com',
    url='https://github.com/doterkuile/open_world_recognition',
    license=license,
    packages=find_packages()
)