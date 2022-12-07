"""An example of code submission for the AutoML Decathlon challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py and metadata together with other necessary files
such as tasks_to_run.yaml, Python modules/packages, pre-trained weights, etc. The final zip file
should not exceed 300MB.

Reference: https://automl.github.io/auto-sklearn/master/
"""

import logging
import math
import sys
import time

## autosklearn
import autosklearn.classification
import autosklearn.regression
import numpy as np
import torch
from torch.utils.data import DataLoader

#

# seeding randomness for reproducibility
np.random.seed(42)
torch.manual_seed(1)


def merge_batches(dataloader: DataLoader, is_single_label: bool):
    x_batches = []
    y_batches = []
    for x, y in dataloader:
        x = x.detach().numpy()
        x = x.reshape(x.shape[0], -1)
        x_batches.append(x)

        y = y.detach().numpy()
        if len(y.shape) > 2:
            y = y.reshape(y.shape[0], -1)

        if is_single_label:
            # for the multi-class, single-label tasks, we need to change the ohe encoding to raw labels for input to training
            y = np.argmax(y, axis=1)

        y_batches.append(y)

    x_matrix = np.concatenate(x_batches, axis=0)
    y_matrix = np.concatenate(y_batches, axis=0)

    return x_matrix, y_matrix


class Model:

    def __init__(self, metadata):
        '''
        The initalization procedure for your method given the metadata of the task
        '''
        """
        Args:
          metadata: an DecathlonMetadata object. Its definition can be found in
              ingestion/dev_datasets.py
        """
        # Attribute necessary for ingestion program to stop evaluation process
        self.done_training = False
        self.metadata_ = metadata
        self.task = self.metadata_.get_dataset_name()
        self.task_type = self.metadata_.get_task_type()

        # Getting details of the data from meta data
        # Product of output dimensions in case of multi-dimensional outputs...
        self.output_dim = math.prod(self.metadata_.get_output_shape())

        self.num_examples_train = self.metadata_.size()

        row_count, col_count = self.metadata_.get_tensor_shape()[2:4]
        channel = self.metadata_.get_tensor_shape()[1]
        sequence_size = self.metadata_.get_tensor_shape()[0]

        self.num_train = self.metadata_.size()
        self.num_test = self.metadata_.get_output_shape()

        # Getting the device available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "Device Found = ", self.device, "\nMoving Model and Data into the device..."
        )
        assert torch.cuda.is_available()  # force xgboost on gpu
        self.input_shape = (channel, sequence_size, row_count, col_count)
        print("\n\nINPUT SHAPE = ", self.input_shape)

        self.model = None

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.total_train_time = 0
        self.total_test_time = 0

        #         # no of examples at each step/batch
        self.train_batch_size = 64
        self.test_batch_size = 64

    def get_dataloader(self, dataset, batch_size, split):
        """Get the PyTorch dataloader. Do not modify this method.
        Args:
          dataset:
          batch_size : batch_size for training set
        Return:
          dataloader: PyTorch Dataloader
        """
        if split == "train":
            dataloader = DataLoader(
                dataset,
                dataset.dataset.required_batch_size or batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=dataset.dataset.collate_fn,
            )
        elif split == "val":
            dataloader = DataLoader(
                dataset,
                dataset.dataset.required_batch_size or batch_size,
                shuffle=False,
                collate_fn=dataset.dataset.collate_fn,
            )
        elif split == "test":
            dataloader = DataLoader(
                dataset,
                dataset.required_batch_size or batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
            )
        return dataloader

    def train(self, dataset, val_dataset=None, val_metadata=None, remaining_time_budget=None):
        '''
        The training procedure of your method given training data, validation data (which is only directly provided in certain tasks, otherwise you are free to create your own validation strategies), and remaining time budget for training.
        '''

        """Train this algorithm on the Pytorch dataset.
        ****************************************************************************
        ****************************************************************************
        Args:
          dataset: a `DecathlonDataset` object. Each of its examples is of the form
                (example, labels)
              where `example` is a dense 4-D Tensor of shape
                (sequence_size, row_count, col_count, num_channels)
              and `labels` is a 1-D or 2-D Tensor
          val_dataset: a 'DecathlonDataset' object. Is not 'None' if a pre-split validation set is provided, in which case you should use it for any validation purposes. Otherwise, you are free to create your own validation split(s) as desired.

          val_metadata: a 'DecathlonMetadata' object, corresponding to 'val_dataset'.
          remaining_time_budget: time remaining to execute train(). The method
              should keep track of its execution time to avoid exceeding its time
              budget. If remaining_time_budget is None, no time budget is imposed.

          remaining_time_budget: the time budget constraint for the task, which may influence the training procedure.
        """
        if remaining_time_budget:
            remaining_time_budget = int(remaining_time_budget)

        if self.metadata_.get_task_type() == "continuous":
            self.model = autosklearn.regression.AutoSklearnRegressor(memory_limit=None,
                                                                     time_left_for_this_task=remaining_time_budget)
        elif self.metadata_.get_task_type() == "single-label":
            self.model = autosklearn.classification.AutoSklearnClassifier(memory_limit=None,
                                                                          time_left_for_this_task=remaining_time_budget)
        elif self.metadata_.get_task_type() == "multi-label":
            self.model = autosklearn.classification.AutoSklearnClassifier(memory_limit=None,
                                                                          time_left_for_this_task=remaining_time_budget)
        else:
            raise NotImplementedError

        # If PyTorch dataloader for training set doen't already exists, get the train dataloader
        if not hasattr(self, "trainloader"):
            self.trainloader = self.get_dataloader(
                dataset,
                self.train_batch_size,
                "train",
            )
        if not hasattr(self, "valloader"):
            self.valloader = self.get_dataloader(
                val_dataset,
                self.train_batch_size,
                "val",
            )

        train_start = time.time()

        # Training (no loop)
        x_train, y_train = merge_batches(self.trainloader, (self.task_type == "single-label"))
        print(x_train.shape, y_train.shape)
        x_valid, y_valid = merge_batches(self.valloader, (self.task_type == "single-label"))

        self.model.fit(
            x_train,
            y_train,
        )

        train_end = time.time()

        train_duration = train_end - train_start
        self.total_train_time += train_duration
        logger.info(
            "{:.2f} sec used for autosklearn. ".format(
                train_duration
            )
            + "Total time used for training: {:.2f} sec. ".format(
                self.total_train_time
            )
        )

    def test_val(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the Pytorch dataloader.
        Args:
          Same as that of `train` method, except that the `labels` will be empty.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
              here `sample_count` is the number of examples in this dataset as test
              set and `output_dim` is the number of labels to be predicted. The
              values should be binary or in the interval [0,1].
          remaining_time_budget: the remaining time budget left for testing, post-training
        """

        test_begin = time.time()

        if not hasattr(self, "valloader"):
            self.valloader = self.get_dataloader(
                dataset,
                self.test_batch_size,
                "val",
            )

        x_test, solutions = merge_batches(self.valloader, (self.task_type == "single-label"))

        # get test predictions from the model
        if self.task_type == "single-label":
            n = self.metadata_.get_output_shape()[0]
            solutions = np.eye(n)[solutions.astype(int)]
            predictions = self.model.predict_proba(x_test)
        elif self.task_type == "multi-label":
            predictions = self.model.predict_proba(x_test)
        else:
            predictions = self.model.predict(x_test)

        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin
        self.total_test_time += test_duration

        return predictions, solutions

    def test(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the Pytorch dataloader.
        Args:
          Same as that of `train` method, except that the `labels` will be empty.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
              here `sample_count` is the number of examples in this dataset as test
              set and `output_dim` is the number of labels to be predicted. The
              values should be binary or in the interval [0,1].
          remaining_time_budget: the remaining time budget left for testing, post-training
        """

        test_begin = time.time()

        logger.info("Begin testing...")

        if not hasattr(self, "testloader"):
            self.testloader = self.get_dataloader(
                dataset,
                self.test_batch_size,
                "test",
            )

        x_test, _ = merge_batches(self.testloader, (self.task_type == "single-label"))

        # get test predictions from the model
        predictions = self.model.predict(x_test)
        # If the task is multi-class single label, the output will be in raw labels; we need to convert to ohe for passing back to ingestion
        if self.task_type == "single-label" or self.task_type == "multi-label":
            predictions = self.model.predict_proba(x_test)
        else:
            predictions = self.model.predict(x_test)

        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin
        self.total_test_time += test_duration

        logger.info(
            "[+] Successfully made one prediction. {:.2f} sec used. ".format(
                test_duration
            )
            + "Total time used for testing: {:.2f} sec. ".format(self.total_test_time)
        )
        return predictions

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################
    @staticmethod
    def is_applicable(metadata):
        is_applicable = False
        timeseries, channel, row, col = metadata.get_tensor_shape()
        input_shape = metadata.get_tensor_shape()
        spacetime_dims = np.count_nonzero(np.array(input_shape)[[0, 2, 3]] != 1)
        dataset_size = metadata.size()
        n_features = timeseries * channel * row * col
        max_features = 50000
        max_dataset_size = 10000
        if dataset_size < max_dataset_size and spacetime_dims <= 1 and n_features < max_features:
            is_applicable = True
        return is_applicable

    def __str__(self):
        return 'autosklearn'


def get_logger(verbosity_level):
    """Set logging format to something like:
    2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(filename)s: %(message)s"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger("INFO")
