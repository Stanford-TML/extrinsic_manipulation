from setuptools import setup, find_packages
from setuptools.command.install import install as InstallCommand


class Install(InstallCommand):
    """Customized setuptools install command which uses pip."""

    def run(self, *args, **kwargs):
        import pip

        pip.main(["install", "."])
        InstallCommand.run(self, *args, **kwargs)


setup(
    name="contact_demo",
    install_requires=[
        "open3d",
        "filterpy",
        "pygame",  # For recording demo
        "drake",
        "easydict",
        "wandb",
        # Below is for IGE
        "gym==0.26.1",
        "torch",
        "omegaconf",
        "termcolor",
        "hydra-core>=1.1",
        "rl-games==1.5.2",
        "pyvirtualdisplay",
    ],
    version="0.0.1a",
    cmdclass={
        "install": Install,
    },
    packages=find_packages(),
)
