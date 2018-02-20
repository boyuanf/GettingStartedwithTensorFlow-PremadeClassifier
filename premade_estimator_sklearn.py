#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import datasets
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from pandas import Series

import argparse
import tensorflow as tf

import iris_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # method 1: using tf to Fetch the data
    #(train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # method 2: using sklearn to Fetch the data
    iris = datasets.load_iris()
    iris_data_X = iris["data"][:]
    iris_data_y = (iris["target"][:]).astype(np.int)
    feature_length = iris_data_X.shape[1]
    data_num = iris_data_X.shape[0]
    iris_data_y = iris_data_y.reshape(-1, 1)

    scaler = StandardScaler()
    scaled_iris_data = scaler.fit_transform(iris_data_X)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=None)
    for train_index, test_index in split.split(scaled_iris_data, iris_data_y):
        train_x = scaled_iris_data[train_index]
        train_y = iris_data_y[train_index]
        test_x = scaled_iris_data[test_index]
        test_y = iris_data_y[test_index]

    # Feature columns describe how to use the input.
    my_feature_columns = []
    train_x_keys = ['SepalLength', 'SepalWidth',
                        'PetalLength', 'PetalWidth']
    for key in train_x_keys:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3)

    train_x_shape = train_x.shape

    # Method 1: convert train_x and train_y to pd. Series
    '''
    # Conver train_x to dict, and key value as Series
    train_x_dict = dict()
    for index, key in enumerate(train_x_keys):
        train_x_dict[key] = Series(train_x[:, index])
    train_y_shape = train_y.shape
    #train_y = Series(train_y.reshape(-1))
    train_y = Series(train_y[:, 0])
    '''
    # Method 2: keep train_x and train_y as array
    train_x_dict = dict()
    test_x_dict = dict()
    for index, key in enumerate(train_x_keys):
        train_x_dict[key] = train_x[:, index]
        test_x_dict[key] = test_x[:, index]

    # Train the Model.
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn_dict(train_x_dict, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn_dict(test_x_dict, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = np.array([[5.1, 3.3, 1.7, 0.5], [5.9, 3.0, 4.2, 1.5], [6.9, 3.1, 5.4, 2.1]])
    scaled_predict_x = scaler.fit_transform(predict_x)
    predict_x_dict = dict()
    for index, key in enumerate(train_x_keys):
        predict_x_dict[key] = scaled_predict_x[:, index]

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn_dict(predict_x_dict,
                                                labels=None,
                                                batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]  # pred_dict is an array contains only one element
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)