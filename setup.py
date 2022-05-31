import setuptools

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

setuptools.setup(
    name="fermionic-operator",
    version="0.1.0",
    description="Fermionic Operator",
    author="Kensuke Tamura",
    install_requires=REQUIREMENTS,
)
