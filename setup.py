from setuptools import setup

setup(name="hockey_analytics",
      version="0.1",
      author="Marshall Mykietyshyn",
      author_email="mmykietyshyn@gmail.com",
      license="MIT",
      description="Ice hockey analytics project.",
      # long_description=open('README.md').read(),
      url="https://github.com/marshineer/Hockey-Analytics",
      packages=["nhl_api"],
      # scripts=['api_scripts/get_season_game_data.py'],
      install_requires=['requests', 'numpy', 'geopy']
      )
