"""sanity tests for the config functions"""
import os
import json
from unittest import TestCase
from tempfile import TemporaryDirectory
from resurfemg.helper_functions.config import Config

class TestConfig(TestCase):

    required_directories = {
        'root_emg_directory',
    }
    required_directories = ['root_emg_directory']

    def test_roots_only(self):
        with TemporaryDirectory() as td:
            same_created_path = os.path.join(td, 'root')
            os.mkdir(same_created_path)
            raw_config = {
                'root_emg_directory': same_created_path,
            }
            config_file = os.path.join(td, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(raw_config, f)

            # for root in self.required_directories:
            #     os.mkdir(os.path.join(td, root))

            config = Config(config_file)
            assert config.get_directory('root_emg_directory')

    def test_missing_config_path(self):
        try:
            Config('non existent')
        except ValueError:
            pass
        else:
            assert False, 'Didn\'t notify on missing config file'
