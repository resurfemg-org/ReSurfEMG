from resurfemg.data_connector.config import Config

config = Config(verbose=True)

base_path = config.get_directory('test_data')
