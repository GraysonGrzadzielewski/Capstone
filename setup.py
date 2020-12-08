from distutils.core import setup

setup(
    name='Using Neural Networks to Develop an AI Capable of Learning to Effectively Play Video Games Repo',
    version='1.0',
    description='Repository for Fall 2020 Capstone',
    author='Brian Crutchley, Grayson Grzadzielewski',
    author_email='bcrutc01@rams.shepherd.edu, ggrzado1@rams.shepherd.edu',
    url='https://github.com/GraysonGrzadzielewski/Capstone',
    long_description=open('README.md','r').read(),
    install_requires=[
        "certifi>=2020.6.20",
        "cloudpickle>=1.6.0",
        "cmake>=3.18.2.post1",
        "cycler>=0.10.0",
        "future>=0.18.2",
        "gym>=0.17.3",
        "gym-retro>=0.8.0",
        "kiwisolver>=1.2.0",
        "matplotlib>=3.3.2",
        "numpy==1.19.2",
        "Pillow>=8.0.0",
        "protobuf>=3.13.0",
        "pyglet>=1.5.0",
        "pyparsing>=2.4.7",
        "python-dateutil>=2.8.1",
        "scipy>=1.5.2",
        "six>=1.15.0",
        "neat-python>=0.92"
    ]
)
