import time

import numpy as np
import torch
from torch.utils.data import random_split

from ask_model import Model as ask_model
from cnn_default_model import Model as cnn_default_model
from cnn_model import Model as cnn_model
from danq import Model as danq_model
from deepsea import Model as deepsea_model
from metrics import scorer
from rnn_model import Model as rnn_model
from unet_default_model import Model as unet_default_model
from unet_model import Model as unet_model
from wrn_model import Model as wrn_model
from xgb_model import Model as xgb_model
from video_model import Model as video_model

np.random.seed(42)
torch.manual_seed(1)


def new_pick_models(metadata):
    model_list = [xgb_model, wrn_model, cnn_model, cnn_default_model, rnn_model, danq_model, deepsea_model, unet_model,
                  unet_default_model, video_model, ask_model]
    return list(filter(lambda x: x.is_applicable(metadata), model_list))


# def get_solution(dataset):
#     dataloader = DataLoader(
#         dataset,
#         dataset.dataset.required_batch_size or 256,
#         shuffle=False,
#         collate_fn=dataset.dataset.collate_fn,
#     )
#     batches = []
#     for _, y in dataloader:
#         batches.append(y.detach().numpy())
#     solution = np.concatenate(batches, axis=0)
#     return solution


class Model:
    def __init__(self, metadata):
        self.metadata_ = metadata
        self.sub_models = new_pick_models(metadata)  # list(map(lambda x: x(metadata), new_pick_models(metadata)))
        self.best_prediction = None
        self.done_training = False
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def train(self, dataset, val_dataset=None, val_metadata=None, remaining_time_budget=None):

        train_size = int(0.7 * len(dataset))
        validation_size = (len(dataset) - train_size) // 2
        test_size = len(dataset) - train_size - validation_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset,
                                                                               [train_size, validation_size, test_size])

    def test(self, predict_dataset, remaining_time_budget=None):
        remaining_time_budget *= 0.85
        start_time = time.time()

        n_models = len(self.sub_models)
        model_budget = remaining_time_budget if n_models == 0 else remaining_time_budget // n_models
        best_score = np.Inf
        # do this in a try catch, make sure that at least one of the models worked
        # if not use wide resnet
        for idx, sub_model in enumerate(self.sub_models):
            current_model = sub_model(self.metadata_)

            train_start = time.time()
            try:
                current_model.train(self.train_dataset, self.val_dataset, None, model_budget)
                prediction, solution = current_model.test_val(self.test_dataset)
                curr_score = scorer(solution, prediction, self.metadata_.get_final_metric())
                print(f'Model {current_model} scored {curr_score} on {self.metadata_.get_dataset_name()}')
                if curr_score < best_score:
                    best_score = curr_score
                    self.best_prediction = current_model.test(predict_dataset, remaining_time_budget)
            except Exception:
                pass

            train_time = time.time() - train_start
            n_remaining_models = n_models - (idx + 1)

            if (n_remaining_models > 0) and (train_time < model_budget):
                additional_budget_per_model = (model_budget - train_time) // n_remaining_models
                model_budget += additional_budget_per_model

        if self.best_prediction is None:
            remaining_time_budget = remaining_time_budget - (time.time() - start_time)
            current_model = xgb_model(self.metadata_)
            current_model.train(self.train_dataset, self.val_dataset, None, remaining_time_budget)
            self.best_prediction = current_model.test(predict_dataset, remaining_time_budget)

        return self.best_prediction
