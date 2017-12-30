# -*- coding: utf-8 -*-
from spiono import setup_spiono

import os
import logging

from dotenv import find_dotenv, load_dotenv
import numpy as np
from sklearn import svm, metrics
from sklearn.externals import joblib

setup_spiono()


def main(features_dir, models_dir):
    """ Runs script to train a model based on train features/targets
     (saved in ../features) and saves serialized model (saved in ../models)
    """
    logger = logging.getLogger(__name__)
    logger.info('Training a model based on training data')

    x_train_file_path = os.path.join(features_dir, 'train_x.csv')
    x_train = np.loadtxt(x_train_file_path, delimiter=',')

    y_train_file_path = os.path.join(features_dir, 'train_y.csv')
    y_train = np.loadtxt(y_train_file_path, delimiter=',')

    clf = svm.SVC()
    clf.fit(x_train, y_train)

    joblib.dump(clf, os.path.join(models_dir, 'classifier.pkl'))

    x_test_file_path = os.path.join(features_dir, 'test_x.csv')
    x_test = np.loadtxt(x_test_file_path, delimiter=',')

    y_test_file_path = os.path.join(features_dir, 'test_y.csv')
    y_test = np.loadtxt(y_test_file_path, delimiter=',')

    y_pred = clf.predict(x_test)

    target_names = ['setosa', 'versicolor', 'virginica']
    print(
        metrics.classification_report(
            y_test, y_pred, target_names=target_names
        )
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    features_dir = os.path.join(project_dir, 'data', 'features')
    models_dir = os.path.join(project_dir, 'models')
    main(features_dir, models_dir)
