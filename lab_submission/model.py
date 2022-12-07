import time
from itertools import combinations

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
from video_model import Model as video_model
from wrn_model import Model as wrn_model
from xgb_fallback_model import Model as xgb_fallback_model
from xgb_model import Model as xgb_model

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

def filter_preds_by_model_idxs(models_info, model_idxs):
    # models_info = [(model_idx, model_preds, solution, score), ...]
    preds = [info[1] for info in filter(lambda x: x[0] in model_idxs, models_info)]
    return preds


def select_ensemble_models(models_info, metadata):
    # models_info = [(model_idx, model_preds, solution, score), ...]

    # First, make sure that all the solutions are the same
    _, _, solution, _ = models_info[0]

    # Disabling this check for now.
    # for _, _, sol, _ in models_info:
    #     assert np.all(sol == solution)

    # Get the model ids
    # model ids are simply the indices of the models in self.sub_models
    model_idxs = [idx for idx, _, _, _ in models_info]

    # Brute force all the combinations
    model_combinations = []
    for r in range(1, len(model_idxs) + 1):
        model_combinations += list(combinations(model_idxs, r=r))

    # Iterate over every combination, average their predictions, score the averaged predictions
    # then update the list of best model idxs
    best_score = np.Inf
    best_model_idxs = None

    print('Model indices being considered for ensembles:', model_idxs)
    print('Number of combinations of models:', len(model_combinations))

    for current_models in model_combinations:
        # E.g, current_models = (0, 1, 3)
        # get the predictions of the models with these ids
        try:
            preds = filter_preds_by_model_idxs(models_info, model_idxs=current_models)
            averaged_preds = average_predictions(preds)

            score = scorer(solution, averaged_preds, metadata.get_final_metric())

            if score < best_score:
                best_score = score
                best_model_idxs = current_models
        except Exception as e:
            print(e)

    if best_model_idxs is not None:
        return best_model_idxs
    else:
        # Return the model with the best score
        best_model = sorted(models_info, key=lambda x: x[3])[0]
        return (best_model[0],)


def average_predictions(predictions):
    assert len(predictions) >= 1, "Predictions list is empty"

    preds = np.array(predictions)
    return np.mean(preds, axis=0)


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
        remaining_time_budget *= 0.8
        start_time = time.time()

        model_test_split_preds = []
        model_final_preds = []

        n_models = len(self.sub_models)
        model_budget = remaining_time_budget if n_models == 0 else remaining_time_budget // n_models
        best_score = np.Inf
        # do this in a try catch, make sure that at least one of the models worked
        # if not use wide resnet
        for idx, sub_model in enumerate(self.sub_models):
            train_start = time.time()
            try:
                instantiation_start = time.time()
                current_model = sub_model(self.metadata_)
                print('Start training: ', current_model)
                instantiation_time = time.time() - instantiation_start
                # Train the model
                current_model.train(self.train_dataset, self.val_dataset, None, model_budget - instantiation_time)

                # Perform inference on the test split created from the train data
                # These predictions will be used to select the models in the ensemble
                # It is critical that the solution returned in every iteration of this loop is the same
                # Or else averaging the predictions makes no sense!
                test_preds, solution = current_model.test_val(self.test_dataset)
                score = scorer(solution, test_preds, self.metadata_.get_final_metric())
                model_test_split_preds.append((idx, test_preds, solution, score))

                # Predict on the final dataset, save the predictions along with the idx of the model
                final_preds = current_model.test(predict_dataset, remaining_time_budget)
                model_final_preds.append(
                    (idx, final_preds, None, None))  # Adding Nones to reuse filter_preds_by_model_idxs
            except Exception as e:
                print(e)

            train_time = time.time() - train_start
            n_remaining_models = n_models - (idx + 1)

            if n_remaining_models > 0:
                additional_budget_per_model = (model_budget - train_time) // n_remaining_models
                model_budget += additional_budget_per_model

        print('Number of successful models:', len(model_final_preds))

        # If all of the models fail, fall back to XGBoost
        if len(model_test_split_preds) == 0:
            print('Falling back to XGBoost because all the models failed!')
            remaining_time_budget = remaining_time_budget - (time.time() - start_time)
            current_model = xgb_fallback_model(self.metadata_)
            current_model.train(self.train_dataset, self.val_dataset, None, remaining_time_budget)
            predictions = current_model.test(predict_dataset, remaining_time_budget)

            return predictions

        # Ensemble selection
        print('Ensemble selection')
        ensemble_model_idxs = select_ensemble_models(model_test_split_preds, self.metadata_)
        print('Selected model indices:', ensemble_model_idxs)
        predictions = filter_preds_by_model_idxs(model_final_preds, ensemble_model_idxs)
        predictions = average_predictions(predictions)  # TODO: convert list of (idx, prediction) to list of predictions

        return predictions
