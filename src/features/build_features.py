# -*- coding: utf-8 -*-
import os
import logging

from dotenv import find_dotenv, load_dotenv
import numpy as np
from sklearn.model_selection import train_test_split


def main(data_dir):
    """
    Runs data processing scripts to turn raw cleaned data
    (saved in ../processed) into features ready to be used
    (saved in ../features).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final features data set from processed data')

    processed_dir_path = os.path.join(data_dir, 'processed')
    data_file_path = os.path.join(processed_dir_path, 'data.csv')
    features = np.loadtxt(data_file_path, delimiter=',', skiprows=1)

    X_train, X_test = train_test_split(
        features, test_size=0.20, shuffle=True, random_state=42
    )

    features_dir_path = os.path.join(data_dir, 'features')

    x_train_file_path = os.path.join(features_dir_path, 'train_x.csv')
    np.savetxt(x_train_file_path, X_train, delimiter=',')

    x_test_file_path = os.path.join(features_dir_path, 'test_x.csv')
    np.savetxt(x_test_file_path, X_test, delimiter=',')

    targets_file_path = os.path.join(processed_dir_path, 'targets.csv')
    targets = np.loadtxt(targets_file_path, delimiter=',', skiprows=1)

    y_train, y_test = train_test_split(
        targets, test_size=0.20, shuffle=True, random_state=42
    )

    y_train_file_path = os.path.join(features_dir_path, 'train_y.csv')
    np.savetxt(y_train_file_path, y_train, delimiter=',', fmt='%.1f')

    y_test_file_path = os.path.join(features_dir_path, 'test_y.csv')
    np.savetxt(y_test_file_path, y_test, delimiter=',', fmt='%.1f')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(os.path.join(project_dir, 'data'))
