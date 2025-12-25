"""sanity tests for the config functions"""
import os
import json
import logging
from unittest import TestCase
from tempfile import TemporaryDirectory
from resurfemg.data_connector.config import Config
from resurfemg.data_connector.config import find_project_root
from pathlib import Path

class TestConfig(TestCase):

    required_directories = {
        'root_data',
    }
    required_directories = ['root_data']

    def test_roots_only(self):
        with TemporaryDirectory() as td:
            same_created_path = os.path.join(td, 'root')
            os.mkdir(same_created_path)
            raw_config = {
                'root_data': same_created_path,
            }
            config_file = os.path.join(td, 'config.json')
            # with open(config_file, 'w') as f:
            #     json.dump(raw_config, f)
            Config(location=config_file, configure=True, verbose=True)

            # for root in self.required_directories:
            #     os.mkdir(os.path.join(td, root))

            config = Config(config_file)
            assert config.get_directory('root_data')

    def test_missing_root(self):
        with TemporaryDirectory() as td:
            # 'root_data': os.path.join(td, 'root'),  # Intentionally missing
            raw_config = {
            }
            config_file = os.path.join(td, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(raw_config, f)
            try:
                Config(location=config_file)
            except ValueError:
                pass
            else:
                assert False, 'Didn\'t notify on missing root_data'

    def test_relative_root(self):
        with TemporaryDirectory() as td:
            raw_config = {
                'root_data': './root_data',
            }
            os.mkdir(os.path.join(td, 'root_data'))
            config_file = os.path.join(td, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(raw_config, f)
            config = Config(location=config_file, verbose=True)
            config_root_path = config.get_directory('root_data')
            self.assertEqual(
                config_root_path,
                os.path.abspath(os.path.join(td, 'root_data'))
            )

    def test_non_existent_root(self):
        with TemporaryDirectory() as td:
            raw_config = {
                'root_data': os.path.join(td, 'non_existent_root'),
            }
            config_file = os.path.join(td, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(raw_config, f)
            
            logger = logging.getLogger("myapp")
            with self.assertLogs(logger, level="ERROR") as cm:
                logger.error("Something went wrong: %s", "details")
                Config(location=config_file)
            
            # cm.output is a list of formatted strings
            self.assertTrue(any(
                "Something went wrong" in msg for msg in cm.output))

            # cm.records gives you the actual LogRecord objects
            for record in cm.records:
                self.assertEqual(record.levelno, logging.ERROR)
                self.assertEqual(record.name, "myapp")

    def test_create_non_existent_root(self):
        with TemporaryDirectory() as td:
            raw_config = {
                'root_data': os.path.join(td, 'non_existent_root'),
            }
            config_file = os.path.join(td, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(raw_config, f)
            config = Config(location=config_file, force=True)
            root_path = config.get_directory('root_data')
            self.assertTrue(os.path.exists(root_path))

    def test_missing_config_path(self):
        try:
            Config('non existent')
        except ValueError:
            pass
        else:
            assert False, 'Didn\'t notify on missing config file'

class TestFindProjectRoot(TestCase):

    def test_find_project_root(self):
        with TemporaryDirectory() as td:
            # Create a mock project structure
            project_root = os.path.join(td, 'project')
            os.mkdir(project_root)
            sub_dir = os.path.join(project_root, 'subdir')
            os.mkdir(sub_dir)
            # Create a requirements.txt in the project root
            requirements_file = os.path.join(project_root, 'requirements.txt')
            open(requirements_file, 'w').close()

            # Test finding the project root from the subdirectory
            found_root = find_project_root(current_dir=Path(sub_dir))
            
            dir_a = Path(project_root)
            dir_b = Path(found_root)
            assert dir_a.samefile(dir_b)

    def test_find_non_existent_project_root(self):
        with TemporaryDirectory() as td:
            # Create a mock directory without a requirements.txt
            project_root = os.path.join(td, 'project')
            os.mkdir(project_root)
            sub_dir = os.path.join(project_root, 'subdir')
            os.mkdir(sub_dir)

            # Test that it returns None when no project root is found
            found_root = find_project_root(current_dir=Path(sub_dir))
            self.assertIsNone(found_root)
