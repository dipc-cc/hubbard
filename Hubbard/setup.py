def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('Hubbard', parent_package, top_path)

    config.add_subpackage('plot')
    config.make_config_py()
    config.add_data_files('EQCONTOUR')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
