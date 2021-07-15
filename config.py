import os
from configparser import ConfigParser


class AppConfig:
    """
    Abstract config class
    Encapsulation working with configs
    """
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    path = 'settings.ini'
    section = 'APP'

    def __init__(self, path: str, section: str):
        self.path = path
        self.section = section

    def __setitem__(self, key: str, value: str):
        """
        Setter for update option configs
        Not create new option if not exist, only update existing

        :param key: str, option name
        :param value: str, data
        :return: None
        """

        # exception if type not string
        if type(value) != str:
            raise TypeError('type must be str')

        config = ConfigParser()
        config.read(f"{self.BASE_PATH}\\{self.path}")

        # exception if not exist
        __ = config[self.section][key]

        config.set(self.section, key, value)

        with open(f"{self.BASE_PATH}/{self.path}", 'w') as config_file:
            config.write(config_file)

    def __getitem__(self, key: str) -> str:
        """
        Return option data

        :param key: str, option name
        :return: str, data from this option
        """

        config = ConfigParser()
        config.read(f"{self.BASE_PATH}/{self.path}")

        return config[self.section][key]

    def get_keys(self) -> list:
        """
        Getting all keys from selected config

        :return: list, collection keys from config
        """

        config = ConfigParser()
        config.read(f"{self.BASE_PATH}/{self.path}")

        return [item[0] for item in config.items(self.section)]

