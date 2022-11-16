from setuptools import setup

setup(name="hockey_analytics",
      version="0.1",
      author="Marshall Mykietyshyn",
      author_email="mmykietyshyn@gmail.com",
      license="MIT",
      description="Ice hockey analytics project.",
      long_description="This projects pulls data from the NHL.com API, creates "
                       "an SQL nhl_database in Postgres using DBeaver, and "
                       "generates analytics based on the available data.",
      url="https://github.com/marshineer/Hockey-Analytics",
      packages=["nhl_api", "nhl_database"],
      install_requires=['requests'])
