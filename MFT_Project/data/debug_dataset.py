import yaml
import qlib
from qlib.data.dataset import Dataset
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.utils import init_instance_by_config

qlib.init(provider_uri='MFT_Project/data', region='cn')
with open('MFT_Project/configs/workflow_config_multi_modal.yaml') as f:
    config = yaml.safe_load(f)

dataset_cfg = config['task']['dataset']
dataset = init_instance_by_config(dataset_cfg, accept_types=Dataset)
pack = dataset.prepare_with_text('train', col_set=DataHandler.CS_RAW, data_key=DataHandlerLP.DK_L)
market_df = pack['market_data']
text_df = pack['text_data']
print('market_df shape:', market_df.shape)
print('market_df columns:', market_df.columns[:10])
print('text_df shape:', None if text_df is None else text_df.shape)
