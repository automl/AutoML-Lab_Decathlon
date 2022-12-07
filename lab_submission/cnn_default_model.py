"""An example of code submission for the AutoML Decathlon challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test'). 

To create a valid submission, zip model.py and metadata together with other necessary files
such as tasks_to_run.yaml, Python modules/packages, pre-trained weights, etc. The final zip file
should not exceed 300MB.

Reference : Resnet - https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
            Regnet - https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_y_16gf.html
"""

import logging
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
# seeding randomness for reproducibility
from torchvision.models import ResNet18_Weights, RegNet_Y_16GF_Weights

from early_stopping_default import EarlyStopping
from lr_finder import LRFinder

np.random.seed(420)
torch.manual_seed(2113)


class PretrainedResNet(nn.Module):
    def __init__(self, input_shape, num_classes, use_pretrained=True):
        super(PretrainedResNet, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.disable_gradients(self.resnet)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        if x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        x = torch.squeeze(x)
        x = self.resnet.forward(x)
        return x

    def disable_gradients(self, model) -> None:
        """
        Freezes the layers of a model
        Args:
            model: The model with the layers to freeze
        Returns:
            None
        """
        for parameter in model.parameters():
            parameter.requires_grad = False


class RegNet(nn.Module):

    def __init__(self, input_shape, num_classes, use_pretrained=True):
        super(RegNet, self).__init__()
        self.net = models.regnet_y_16gf(weights=RegNet_Y_16GF_Weights.DEFAULT)
        # self.disable_gradients(self.net)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        if x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        x = torch.squeeze(x)
        x = self.net.forward(x)
        return x

    def disable_gradients(self, model) -> None:
        """
        Freezes the layers of a model
        Args:
            model: The model with the layers to freeze
        Returns:
            None
        """
        for parameter in model.parameters():
            parameter.requires_grad = False


# PyTorch Model class
class LinearModel(nn.Module):
    """
    Defines a module that will be created in '__init__' of the 'Model' class below, and will be used for training and predictions.
    """

    def __init__(self, input_shape, output_dim):
        """a simple linear model"""
        super(LinearModel, self).__init__()

        fc_size = np.prod(input_shape)
        print("input_shape, fc_size", input_shape, fc_size)
        self.fc = nn.Linear(fc_size, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Model:
    def __init__(self, metadata):
        """
        The initalization procedure for your method given the metadata of the task
        """
        """
        Args:
          metadata: an DecathlonMetadata object. Its definition can be found in
              ingestion/dev_datasets.py
        """
        self.metadata_ = metadata

        # Getting details of the data from meta data
        # Product of output dimensions in case of multi-dimensional outputs...
        self.output_dim = math.prod(self.metadata_.get_output_shape())

        row_count, col_count = self.metadata_.get_tensor_shape()[2:4]
        channel = self.metadata_.get_tensor_shape()[1]
        sequence_size = self.metadata_.get_tensor_shape()[0]

        self.num_train = self.metadata_.size()

        # Getting the device available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "Device Found = ", self.device, "\nMoving Model and Data into the device..."
        )

        self.input_shape = (sequence_size, channel, row_count, col_count)
        print("\n\nINPUT SHAPE = ", self.input_shape)

        # getting an object for the PyTorch Model class for Model Class
        # use CUDA if available
        self.model = RegNet(self.input_shape, self.output_dim)  # maybe change output_dim
        # print(self.model)
        self.model.to(self.device)

        # PyTorch Optimizer and Criterion
        if self.metadata_.get_task_type() == "continuous":
            self.criterion = nn.MSELoss()
        elif self.metadata_.get_task_type() == "single-label":
            self.criterion = nn.CrossEntropyLoss()
        elif self.metadata_.get_task_type() == "multi-label":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.total_train_time = 0
        self.total_test_time = 0

        # no of examples at each step/batch
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

    def train(
            self, dataset, val_dataset=None, val_metadata=None, remaining_time_budget=None
    ):
        """
        The training procedure of your method given training data, validation data (which is only directly provided in certain tasks, otherwise you are free to create your own validation strategies), and remaining time budget for training.
        """

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
              should be tuned to fit within this budget.
        """

        logger.info("Begin training...")

        # If PyTorch dataloader for training set doen't already exists, get the train dataloader
        if not hasattr(self, "trainloader") or not hasattr(self, "valloader"):
            self.trainloader = self.get_dataloader(
                dataset,
                self.train_batch_size,
                "train",
            )
            self.valloader = self.get_dataloader(
                val_dataset,
                self.train_batch_size,
                "val",
            )

        train_start = time.time()

        # Training loop
        self.trainloop(self.criterion, self.optimizer, time_limit=remaining_time_budget)
        train_end = time.time()

        # Update for time budget managing
        train_duration = train_end - train_start
        self.total_train_time += train_duration

        logger.info(
            "Training finished. {:.2f} sec used. ".format(train_duration)
            + "Total time used for training: {:.2f} sec. ".format(self.total_train_time)
        )

    def test_val(self, dataset, remaining_time_budget=None):
        test_begin = time.time()

        if not hasattr(self, "valloader"):
            self.valloader = self.get_dataloader(
                dataset,
                self.test_batch_size,
                "val",
            )

        # get predictions from the test loop
        predictions, solutions = self.testloop(self.valloader)

        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin

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
        """

        test_begin = time.time()

        logger.info("Begin testing...")

        if not hasattr(self, "testloader"):
            self.testloader = self.get_dataloader(
                dataset,
                self.test_batch_size,
                "test",
            )

        # get predictions from the test loop
        predictions, _ = self.testloop(self.testloader)

        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin

        logger.info(
            "[+] Successfully made predictions. {:.2f} sec used. ".format(test_duration)
        )
        return predictions

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################

    def trainloop(self, criterion, optimizer, time_limit):
        """Training loop with no of given steps
        Args:
          criterion: PyTorch Loss function
          Optimizer: PyTorch optimizer for training
          epochs: No of epochs to train the model

        Return:
          None, updates the model parameters
        """

        early_stopping = EarlyStopping(metadata=self.metadata_, run_time=time_limit)
        running = True
        epoch = 0
        while running:
            epoch += 1
            self.model.train()
            for images, labels in self.trainloader:
                images = images.float().to(self.device)
                labels = labels.float().to(self.device)
                optimizer.zero_grad()

                logits = self.model(images)
                loss = criterion(logits, labels.reshape(labels.shape[0], -1))

                if hasattr(self, "scheduler"):
                    self.scheduler.step(loss)

                loss.backward()
                optimizer.step()
            val_score = self.val_score(criterion)
            running = early_stopping.running(val_score)
        print(f'epochs: {epoch}')

    def val_score(self, criterion):
        val_score = 0.
        with torch.no_grad():
            self.model.eval()
            for val_images, val_labels in iter(self.valloader):
                val_images = val_images.float().to(self.device)
                val_labels = val_labels.float().to(self.device)
                val_logits = self.model(val_images)
                val_score += criterion(val_logits, val_labels.reshape(val_labels.shape[0], -1))
        return val_score

    def testloop(self, dataloader):
        """
        Args:
          dataloader: PyTorch test dataloader

        Return:
          preds: Predictions of the model as Numpy Array.
        """
        preds = []
        solutions = []
        with torch.no_grad():
            self.model.eval()
            for images, target in iter(dataloader):
                if torch.cuda.is_available():
                    images = images.float().cuda()
                else:
                    images = images.float()
                logits = self.model(images)

                # Choose correct prediction type
                if self.metadata_.get_task_type() == "continuous":
                    pred = logits
                elif self.metadata_.get_task_type() == "single-label":
                    pred = torch.softmax(logits, dim=1).data
                elif self.metadata_.get_task_type() == "multi-label":
                    pred = torch.sigmoid(logits).data
                else:
                    raise NotImplementedError

                solutions.append(target.cpu().numpy())
                preds.append(pred.cpu().numpy())

        preds = np.vstack(preds)
        solutions = np.vstack(solutions)
        return preds, solutions

    @staticmethod
    def is_applicable(metadata):
        is_applicable = False

        input_shape = metadata.get_tensor_shape()
        timeseries, channel, row, col = metadata.get_tensor_shape()
        spacetime_dims = np.count_nonzero(np.array(input_shape)[[0, 2, 3]] != 1)
        output_shape = metadata.get_output_shape()

        if spacetime_dims == 2: # for images
            is_applicable = True
            if len(output_shape) > 1: # where the output is not an image
                if output_shape[0] == row and output_shape[1] == col:
                    is_applicable = False

        return is_applicable

    def __str__(self):
        return 'CNN'


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
