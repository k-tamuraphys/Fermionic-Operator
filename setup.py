import setuptools

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

setuptools.setup(
    name="fermionic-operator",
    version="0.0.1",
    description="Fermionic Operator",
    author="Kensuke Tamura",
    install_requires=REQUIREMENTS,
    packages=setuptools.find_packages(),
)
