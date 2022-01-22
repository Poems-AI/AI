__all__ = ['get_config_value', 'set_config_value']


config = {
    'KAGGLE_DS_ROOT': '/kaggle/input',
    'OWN_DS_ROOT': './dataset',
    'RNG_SEED': 42
}


def set_config_value(key, value):
    assert key in config, 'Invalid config key'
    config[key] = value


def get_config_value(key):
    assert key in config, 'Invalid config key'
    return config[key]
