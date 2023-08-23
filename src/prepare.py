import os
import datetime

from src.helpers import prepare_data


if __name__ == '__main__':
    mode = os.getenv('mode')
    root_dir = '/srv/data'
    if mode == 'train':
        prepare_data(root_dir, 'ml_100k.zip')
    else:
        raise ValueError('Invalid mode value')