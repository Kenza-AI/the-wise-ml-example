# -*- coding: utf-8 -*-
import os
import logging

from dotenv import find_dotenv, load_dotenv
import numpy as np
from sklearn import datasets


def main(data_dir):
    """ Runs data processing scripts to turn raw data into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    iris = datasets.load_iris()
    iris_data = iris.data
    iris_targets = iris.target

    output_dir = os.path.join(data_dir, 'processed')
    data_file_path = os.path.join(output_dir, 'data.csv')
    data_headers = ','.join(iris.feature_names)
    np.savetxt(
        data_file_path,
        iris_data,
        header=data_headers,
        delimiter=',',
        comments=''
    )

    targets_file_path = os.path.join(output_dir, 'targets.csv')
    np.savetxt(
        targets_file_path,
        iris_targets,
        header='target',
        delimiter=',',
        fmt='%.1f',
        comments=''
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(os.path.join(project_dir, 'data'))
