"""
Neuraxle's AutoML Classes
====================================
Classes used to build any Automatic Machine Learning strategies.

..
    Copyright 2019, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""

import copy
import gc
import glob
import hashlib
import json
import logging
import math
import multiprocessing
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Callable, List, Union, Tuple

import numpy as np

from neuraxle.base import BaseStep, ExecutionContext, ForceHandleMixin, ExecutionPhase, _HasChildrenMixin, \
    LOGGER_FORMAT, DATE_FORMAT
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.metaopt.callbacks import BaseCallback, CallbackList, ScoringCallback
from neuraxle.metaopt.observable import _Observable, _Observer
from neuraxle.metaopt.random import BaseCrossValidationWrapper
from neuraxle.metaopt.trial import Trial, TrialSplit, TRIAL_STATUS, Trials


class HyperparamsRepository(_Observable[Tuple['HyperparamsRepository', Trial]], ABC):
    """
    Hyperparams repository that saves hyperparams, and scores for every AutoML trial.

    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self, hyperparameter_selection_strategy=None, cache_folder=None, best_retrained_model_folder=None):
        super().__init__()
        if cache_folder is None:
            cache_folder = 'trials'
        if best_retrained_model_folder is None:
            best_retrained_model_folder = os.path.join(cache_folder, 'best')
        self.best_retrained_model_folder = best_retrained_model_folder

        self.hyperparameter_selection_strategy = hyperparameter_selection_strategy
        self.cache_folder = cache_folder

    def set_strategy(self, hyperparameter_selection_strategy: 'BaseHyperparameterSelectionStrategy'):
        """
        Set hyperparameter selection strategy.

        :param hyperparameter_selection_strategy: hyperparameter selection strategy.
        :return:
        """
        self.hyperparameter_selection_strategy = hyperparameter_selection_strategy

    @abstractmethod
    def load_all_trials(self, status: 'TRIAL_STATUS') -> 'Trials':
        """
        Load all hyperparameter trials with their corresponding score.
        Sorted by creation date.

        :return: Trials (hyperparams, scores)
        """
        pass

    def save_trial(self, trial: 'Trial'):
        """
        Save trial, and notify trial observers.

        :param trial: trial to save.
        :return:
        """
        self._save_trial(trial)
        self.on_next(value=(self, trial))  # notify a tuple of (repo, trial) to observers

    @abstractmethod
    def _save_trial(self, trial: 'Trial'):
        """
        save trial.

        :param trial: trial to save.
        :return:
        """
        pass

    def get_best_hyperparams(self) -> HyperparameterSamples:
        """
        Get best hyperparams from all of the saved trials.

        :return: best hyperparams.
        """
        trials = self.load_all_trials(status=TRIAL_STATUS.SUCCESS)
        best_hyperparams = HyperparameterSamples(trials.get_best_hyperparams())
        return best_hyperparams

    def _save_best_model(self, step: BaseStep, trial_hash: str):
        step.set_name(trial_hash).save(ExecutionContext(self.best_retrained_model_folder), full_dump=True)

    def _load_best_model(self, trial_hash: str) -> BaseStep:
        return ExecutionContext(str(self.best_retrained_model_folder)).load(trial_hash)

    def get_best_model(self):
        """
        Load the best model saved inside the best retrained model folder.

        :return:
        """
        hyperparams: HyperparameterSamples = self.get_best_hyperparams()
        trial_hash: str = self._get_trial_hash(HyperparameterSamples(hyperparams).to_flat_dict())
        p: BaseStep = self._load_best_model(trial_hash)
        return p

    def save_best_model(self, step: BaseStep):
        """
        Save the best model inside the best retrained model folder.

        :param step: step to save
        :return: saved step
        """
        hyperparams = step.get_hyperparams().to_flat_dict()
        trial_hash = self._get_trial_hash(hyperparams)
        self._save_best_model(step, trial_hash)
        return step

    def new_trial(self, auto_ml_container: 'AutoMLContainer'):
        """
        Create a new trial with the best next hyperparams.

        :param context:
        :param auto_ml_container: auto ml data container
        :return: trial
        """
        hyperparams = self.hyperparameter_selection_strategy.find_next_best_hyperparams(auto_ml_container)
        logger = self._create_logger_for_trial(auto_ml_container.trial_number)
        logger.info('\nnew trial: {}'.format(json.dumps(hyperparams.to_nested_dict(), sort_keys=True, indent=4)))

        trial = Trial(
            hyperparams=hyperparams,
            save_trial_function=self.save_trial,
            logger=logger,
            cache_folder=self.cache_folder,
            main_metric_name=auto_ml_container.main_scoring_metric_name
        )
        return trial


    def _get_trial_hash(self, hp_dict):
        """
        Hash hyperparams with md5 to create a trial hash.

        :param hp_dict:
        :return:
        """
        current_hyperparameters_hash = hashlib.md5(str.encode(str(hp_dict))).hexdigest()
        return current_hyperparameters_hash

    def _create_logger_for_trial(self, trial_number) -> logging.Logger:

        os.makedirs(self.cache_folder, exist_ok=True)

        logfile_path = os.path.join(self.cache_folder, f"trial_{trial_number}.log")
        logger_name = f"trial_{trial_number}"
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(fmt=LOGGER_FORMAT, datefmt=DATE_FORMAT)
        file_handler = logging.FileHandler(filename=logfile_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger


class InMemoryHyperparamsRepository(HyperparamsRepository):
    """
    In memory hyperparams repository that can print information about trials.
    Useful for debugging.

    Example usage :

    .. code-block:: python

        InMemoryHyperparamsRepository(
            hyperparameter_selection_strategy=RandomSearchHyperparameterSelectionStrategy(),
            print_func=print,
            cache_folder='cache',
            best_retrained_model_folder='best'
        )

    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self, hyperparameter_selection_strategy=None, cache_folder: str = None,
                 best_retrained_model_folder=None):
        HyperparamsRepository.__init__(
            self,
            hyperparameter_selection_strategy=hyperparameter_selection_strategy,
            cache_folder=cache_folder,
            best_retrained_model_folder=best_retrained_model_folder
        )
        self.cache_folder = cache_folder

        self.trials = Trials()

    def load_all_trials(self, status: 'TRIAL_STATUS' = None) -> 'Trials':
        """
        Load all trials with the given status.

        :param status: trial status
        :return: list of trials
        """
        return self.trials.filter(status)

    def _save_trial(self, trial: 'Trial'):
        """
        Save trial.

        :param trial: trial to save
        :return:
        """
        self.trials.append(trial)


class HyperparamsJSONRepository(HyperparamsRepository):
    """
    Hyperparams repository that saves json files for every AutoML trial.

    Example usage :

    .. code-block:: python

        HyperparamsJSONRepository(
            hyperparameter_selection_strategy=RandomSearchHyperparameterSelectionStrategy(),
            cache_folder='cache',
            best_retrained_model_folder='best'
        )


    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.trial.HyperparameterSamples`
    """

    def __init__(
            self,
            hyperparameter_selection_strategy: 'BaseHyperparameterSelectionStrategy' = None,
            cache_folder=None,
            best_retrained_model_folder=None
    ):
        HyperparamsRepository.__init__(
            self,
            hyperparameter_selection_strategy=hyperparameter_selection_strategy,
            cache_folder=cache_folder,
            best_retrained_model_folder=best_retrained_model_folder
        )
        self.json_path_remove_on_update = None

    def _save_trial(self, trial: 'Trial'):
        """
        Save trial json.

        :param trial: trial to save
        :return:
        """
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        self._remove_previous_trial_state_json()

        trial_path_func = {
            TRIAL_STATUS.SUCCESS: self._get_successful_trial_json_file_path,
            TRIAL_STATUS.FAILED: self._get_failed_trial_json_file_path,
            TRIAL_STATUS.STARTED: self._get_ongoing_trial_json_file_path,
            TRIAL_STATUS.PLANNED: self._get_new_trial_json_file_path
        }
        trial_file_path = trial_path_func[trial.status](trial)

        with open(trial_file_path, 'w+') as outfile:
            json.dump(trial.to_json(), outfile)

        if trial.status in (TRIAL_STATUS.SUCCESS, TRIAL_STATUS.FAILED):
            self.json_path_remove_on_update = None
        else:
            self.json_path_remove_on_update = trial_file_path

        # Sleeping to have a valid time difference between files when reloading them to sort them by creation time:
        time.sleep(0.1)

    def new_trial(self, auto_ml_container: 'AutoMLContainer'):
        """
        Create new hyperperams trial json file.

        :param auto_ml_container: auto ml container
        :return:
        """
        trial = HyperparamsRepository.new_trial(self, auto_ml_container)
        self._save_trial(trial)

        return trial

    def load_all_trials(self, status: 'TRIAL_STATUS' = None) -> 'Trials':
        """
        Load all hyperparameter trials with their corresponding score.
        Reads all the saved trial json files, sorted by creation date.

        :return: (hyperparams, scores)
        """
        trials = Trials()

        files = glob.glob(os.path.join(self.cache_folder, '*.json'))

        # sort by created date:
        def getmtimens(filename):
            return os.stat(filename).st_mtime_ns

        files.sort(key=getmtimens)

        for base_path in files:
            with open(base_path) as f:
                try:
                    trial_json = json.load(f)
                except Exception as err:
                    print('invalid trial json file'.format(base_path))
                    print(traceback.format_exc())
                    continue

            if status is None or trial_json['status'] == status.value:
                trials.append(Trial.from_json(
                    update_trial_function=self.save_trial,
                    trial_json=trial_json
                ))

        return trials

    def _get_successful_trial_json_file_path(self, trial: 'Trial') -> str:
        """
        Get the json path for the given successful trial.

        :param trial: trial
        :return: str
        """
        trial_hash = self._get_trial_hash(trial.hyperparams.to_flat_dict())
        return os.path.join(
            self.cache_folder,
            str(float(trial.get_validation_score())).replace('.', ',') + "_" + trial_hash
        ) + '.json'

    def _get_failed_trial_json_file_path(self, trial: 'Trial'):
        """
        Get the json path for the given failed trial.

        :param trial: trial
        :return: str
        """
        trial_hash = self._get_trial_hash(trial.hyperparams.to_flat_dict())
        return os.path.join(self.cache_folder, 'FAILED_' + trial_hash) + '.json'

    def _get_ongoing_trial_json_file_path(self, trial: 'Trial'):
        """
        Get ongoing trial json path.
        """
        hp_dict = trial.hyperparams.to_flat_dict()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)
        return os.path.join(self.cache_folder, "ONGOING_" + current_hyperparameters_hash) + '.json'

    def _get_new_trial_json_file_path(self, trial: 'Trial'):
        """
        Get new trial json path.
        """
        hp_dict = trial.hyperparams.to_flat_dict()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)
        return os.path.join(self.cache_folder, "NEW_" + current_hyperparameters_hash) + '.json'

    def _remove_previous_trial_state_json(self):
        if self.json_path_remove_on_update and os.path.exists(self.json_path_remove_on_update):
            os.remove(self.json_path_remove_on_update)

    def subscribe_to_cache_folder_changes(self, refresh_interval_in_seconds: int,
                                          observer: _Observer[Tuple[HyperparamsRepository, Trial]]):
        """
        Every refresh_interval_in_seconds

        :param refresh_interval_in_seconds: number of seconds to wait before sending updates to the observers
        :param observer:
        :return:
        """
        self._observers.add(observer)
        # TODO: start a process that notifies observers anytime a the file of a trial changes
        # possibly use this ? https://github.com/samuelcolvin/watchgod
        # note: this is how you notify observers self.on_next((self, updated_trial))


class BaseHyperparameterSelectionStrategy(ABC):
    @abstractmethod
    def find_next_best_hyperparams(self, auto_ml_container: 'AutoMLContainer') -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials.

        :param auto_ml_container: trials data container
        :return: next best hyperparams
        """
        raise NotImplementedError()


class Trainer:
    """

    Example usage :

    .. code-block:: python

        trainer = Trainer(
            epochs=10,
            callbacks=[EarlyStoppingCallback()],
            scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
            validation_splitter=ValidationSplitter(test_size=0.15),
            print_func=print
        )

        repo_trial = trainer.train(
            pipeline=pipeline,
            data_inputs=data_inputs,
            expected_outputs=expected_outputs
        )


    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(
            self,
            epochs: int,
            scoring_callback: ScoringCallback,
            validation_splitter: 'BaseValidationSplitter',
            callbacks: List[BaseCallback] = None,
            hyperparams_repository: HyperparamsRepository = None
    ):
        self.epochs: int = epochs
        self.validation_split_function = validation_splitter

        if callbacks is None:
            callbacks = []
        callbacks: List[BaseCallback] = [scoring_callback] + callbacks
        self.callbacks: CallbackList = CallbackList(callbacks)

        if hyperparams_repository is None:
            hyperparams_repository = InMemoryHyperparamsRepository()
        self.hyperparams_repository: HyperparamsRepository = hyperparams_repository

    def train(self, pipeline: BaseStep, data_inputs, expected_outputs=None, context: ExecutionContext = None) -> Trial:
        """
        Train pipeline using the validation splitter.
        Track training, and validation metrics for each epoch.
        Note: the present method is just a shortcut to using the `execute_trial` method with less boilerplate code needed. Refer to `execute_trial` for full flexibility

        :param pipeline: pipeline to train on
        :param data_inputs: data inputs
        :param expected_outputs: expected ouptuts to fit on
        :return: executed trial

        """
        assert not (context is None)  # TODO: change order of arguments so that context isn't an optional argument

        validation_splits: List[
            Tuple[DataContainer, DataContainer]] = self.validation_split_function.split_data_container(
            DataContainer(data_inputs=data_inputs, expected_outputs=expected_outputs),
            context=context
        )

        repo_trial: Trial = Trial(
            pipeline=pipeline,
            logger=context.logger,
            hyperparams=pipeline.get_hyperparams(),
            main_metric_name=self.get_main_metric_name(),
            save_trial_function=self.hyperparams_repository.save_trial
        )

        self.execute_trial(
            pipeline=pipeline,
            trial_number=1,
            repo_trial=repo_trial,
            context=context,
            validation_splits=validation_splits,
            n_trial=1,
            delete_pipeline_on_completion=False
        )

        return repo_trial

    def execute_trial(
            self,
            pipeline: BaseStep,
            trial_number: int,
            repo_trial: Trial,
            context: ExecutionContext,
            validation_splits: List[Tuple[DataContainer, DataContainer]],
            n_trial: int,
            delete_pipeline_on_completion: bool = True
    ):
        """
        Train pipeline using the validation splitter.
        Track training, and validation metrics for each epoch.

        :param pipeline: pipeline to train on
        :param trial_number: trial number
        :param repo_trial: repo trial
        :param validation_splits: validation splits
        :param context: execution context
        :param n_trial: total number of trials that will be executed
        :param delete_pipeline_on_completion: bool to delete pipeline on completion or not
        :return: executed trial split
        """
        for training_data_container, validation_data_container in validation_splits:
            p = copy.deepcopy(pipeline)
            p.update_hyperparams(repo_trial.hyperparams)
            repo_trial.set_hyperparams(p.get_hyperparams())

            repo_trial_split: TrialSplit = repo_trial.new_validation_split(
                pipeline=p,
                delete_pipeline_on_completion=delete_pipeline_on_completion
            )

            with repo_trial_split:
                trial_split_description = _get_trial_split_description(
                    repo_trial=repo_trial,
                    repo_trial_split_number=repo_trial_split.split_number,
                    validation_splits=validation_splits,
                    trial_number=trial_number,
                    n_trial=n_trial
                )

                context.logger.info('fitting trial {}'.format(
                    trial_split_description
                ))

                repo_trial_split = self.fit_trial_split(
                    trial_split=repo_trial_split,
                    train_data_container=training_data_container,
                    validation_data_container=validation_data_container,
                    context=context
                )

                repo_trial_split.set_success()

                context.logger.info('success trial {}\nbest score: {} at epoch {}'.format(
                    trial_split_description,
                    repo_trial_split.get_best_validation_score(),
                    repo_trial_split.get_n_epochs_to_best_validation_score()
                ))

        return repo_trial_split

    def fit_trial_split(
            self,
            trial_split: TrialSplit,
            train_data_container: DataContainer,
            validation_data_container: DataContainer,
            context: ExecutionContext
    ) -> TrialSplit:
        """
        Train pipeline using the training data container.
        Track training, and validation metrics for each epoch.

        :param train_data_container: train data container
        :param validation_data_container: validation data container
        :param trial_split: trial to execute
        :param context: execution context

        :return: executed trial
        """

        for i in range(self.epochs):
            context.logger.info('epoch {}/{}'.format(i + 1, self.epochs))
            trial_split = trial_split.fit_trial_split(train_data_container.copy(),
                                                      context.copy().set_execution_phase(ExecutionPhase.TRAIN))
            y_pred_train = trial_split.predict_with_pipeline(train_data_container.copy(),
                                                             context.copy().set_execution_phase(
                                                                 ExecutionPhase.VALIDATION))
            y_pred_val = trial_split.predict_with_pipeline(validation_data_container.copy(),
                                                           context.copy().set_execution_phase(
                                                               ExecutionPhase.VALIDATION))

            if self.callbacks.call(
                    trial_split=trial_split,
                    epoch_number=i,
                    total_epochs=self.epochs,
                    input_train=train_data_container,
                    pred_train=y_pred_train,
                    input_val=validation_data_container,
                    pred_val=y_pred_val,
                    context=context.copy().set_execution_phase(ExecutionPhase.VALIDATION),
                    is_finished_and_fitted=False,
            ):
                break
            # Saves the metrics
            trial_split.save_parent_trial()

        return trial_split

    def refit(self, p: BaseStep, data_container: DataContainer, context: ExecutionContext) -> BaseStep:
        """
        Refit the pipeline on the whole dataset (without any validation technique).

        :param p: trial to refit
        :param data_container: data container
        :param context: execution context

        :return: fitted pipeline
        """
        context.set_execution_phase(ExecutionPhase.TRAIN)
        for i in range(self.epochs):
            p = p.handle_fit(data_container, context)

        return p

    def get_main_metric_name(self) -> str:
        """
        Get main metric name.

        :return:
        """
        return self.callbacks[0].name


class AutoML(ForceHandleMixin, _HasChildrenMixin, BaseStep):
    """
    A step to execute any Automatic Machine Learning Algorithms.

    Example usage :

    .. code-block:: python

        auto_ml = AutoML(
            pipeline,
            n_trials=n_iter,
            validation_split_function=validation_splitter(0.2),
            hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
            scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
            callbacks=[
                MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False)
            ],
            refit_trial=True,
            cache_folder_when_no_handle=str(tmpdir)
        )

        auto_ml = auto_ml.fit(data_inputs, expected_outputs)


    .. seealso::
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.base.ForceHandleOnlyMixin`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.trial.Trials`,
        :class:`HyperparamsRepository`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`BaseValidationSplitter`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(
            self,
            pipeline: BaseStep,
            validation_splitter: 'BaseValidationSplitter',
            refit_trial: bool,
            scoring_callback: ScoringCallback,
            hyperparams_optimizer: BaseHyperparameterSelectionStrategy = None,
            hyperparams_repository: HyperparamsRepository = None,
            n_trials: int = 10,
            epochs: int = 1,
            callbacks: List[BaseCallback] = None,
            refit_scoring_function: Callable = None,
            cache_folder_when_no_handle=None,
            n_jobs=None,
            continue_loop_on_error=True
    ):
        """
        Notes on multiprocess :
              Usage of a multiprocess-safe hyperparams repository is recommended, although it is, most of the time, not necessary.
              Beware of the behaviour of HyperparamsRepository's observers/subscribers.
              Context instances are not shared between trial but copied. So is the AutoML loop and the DataContainers.


        :param pipeline: The pipeline, or BaseStep, which will be use by the AutoMLloop
        :param validation_splitter: A :class:`BaseValidationSplitter` instance to split data between training and validation set.
        :param refit_trial: A boolean indicating whether to perform, after ,  a fit call with
        :param scoring_callback: The scoring callback to use during training
        :param hyperparams_optimizer: a :class:`BaseHyperparameterSelectionStrategy` instance that can be queried for new sets of hyperparameters.
        :param hyperparams_repository: a :class:`HyperparamsRepository` instance to store experiement status and results.
        :param n_trials: The number of different hyperparameters to try.
        :param epochs: The number of epoch to perform for each trial.
        :param callbacks: A list of callbacks to perform after each epoch.
        :param refit_scoring_function: A scoring function to use on a refit call
        :param cache_folder_when_no_handle: default cache folder used if auto_ml_loop isn't called through handler functions.
        :param n_jobs: If n_jobs in (None, 1), then automl is executed in a single process, which may spawns on multiple thread. if n_jobs > 1, then n_jobs process are launched, if n_jobs <= -1 then (n_cpus + 1 + n_jobs) process are launched. One trial at a time is executed by process.
        :param continue_loop_on_error:
        """
        BaseStep.__init__(self)
        ForceHandleMixin.__init__(self, cache_folder=cache_folder_when_no_handle)

        self.validation_splitter: BaseValidationSplitter = validation_splitter

        if hyperparams_optimizer is None:
            hyperparams_optimizer = RandomSearchHyperparameterSelectionStrategy()
        self.hyperparameter_optimizer: BaseHyperparameterSelectionStrategy = hyperparams_optimizer

        if hyperparams_repository is None:
            hyperparams_repository = HyperparamsJSONRepository(hyperparams_optimizer, cache_folder_when_no_handle)
        else:
            hyperparams_repository.set_strategy(hyperparams_optimizer)

        self.pipeline: BaseStep = pipeline

        self.n_trial: int = n_trials
        self.hyperparams_repository: HyperparamsRepository = hyperparams_repository

        self.refit_scoring_function: Callable = refit_scoring_function

        self.refit_trial: bool = refit_trial
        self.n_jobs = n_jobs

        self.error_types_to_raise = (SystemError, SystemExit, EOFError, KeyboardInterrupt) if continue_loop_on_error \
            else (Exception,)

        self.trainer = Trainer(
            epochs=epochs,
            scoring_callback=scoring_callback,
            callbacks=callbacks,
            validation_splitter=validation_splitter
        )

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseStep':
        """
        Run Auto ML Loop.
        Find the best hyperparams using the hyperparameter optmizer.
        Evaluate the pipeline on each trial using a validation technique.

        :param data_container: data container to fit
        :param context: execution context

        :return: self
        """
        validation_splits = self.validation_splitter.split_data_container(
            data_container=data_container,
            context=context
        )

        # Keeping a reference of the main logger
        main_logger = context.logger

        if self.n_jobs in (None, 1):
            # Single Process
            for trial_number in range(self.n_trial):
                self._attempt_trial(trial_number, validation_splits, context)
        else:
            # Multiprocssing
            context.logger.info(f"Number of processors available: {multiprocessing.cpu_count()}")

            if isinstance(self.hyperparams_repository, InMemoryHyperparamsRepository):
                raise ValueError(
                    "Cannot use InMemoryHyperparamsRepository for multiprocessing, use json-based repository.")

            n_jobs = self.n_jobs
            if n_jobs <= -1:
                n_jobs = multiprocessing.cpu_count() + 1 + self.n_jobs

            with multiprocessing.get_context("spawn").Pool(processes=n_jobs) as pool:
                # Note : In python 3.6, the root logger possess a lock which make it non-pickable.
                # The following line is not necessary in Python 3.7 and 3.8.
                context.set_logger(None)

                args = [(self, trial_number, validation_splits, context) for trial_number in range(self.n_trial)]
                pool.starmap(AutoML._attempt_trial, args)

        context.set_logger(main_logger)

        best_hyperparams = self.hyperparams_repository.get_best_hyperparams()

        context.logger.info(
            '\nbest hyperparams: {}'.format(json.dumps(best_hyperparams.to_nested_dict(), sort_keys=True, indent=4)))

        # Notify HyperparamsRepository subscribers
        self.hyperparams_repository.on_complete(value=self.hyperparams_repository)

        if self.refit_trial:
            p: BaseStep = self._load_virgin_model(hyperparams=best_hyperparams)
            p = self.trainer.refit(
                p=p,
                data_container=data_container,
                context=context.set_execution_phase(ExecutionPhase.TRAIN)
            )

            self.hyperparams_repository.save_best_model(p)

        return self

    def _attempt_trial(self, trial_number, validation_splits, context: ExecutionContext):

        try:
            auto_ml_data = AutoMLContainer(
                trial_number=trial_number,
                trials=self.hyperparams_repository.load_all_trials(TRIAL_STATUS.SUCCESS),
                hyperparameter_space=self.pipeline.get_hyperparams_space(),
                main_scoring_metric_name=self.trainer.get_main_metric_name()
            )

            with self.hyperparams_repository.new_trial(auto_ml_data) as repo_trial:
                repo_trial_split = None
                context.set_logger(repo_trial.logger)
                context.logger.info('trial {}/{}'.format(trial_number + 1, self.n_trial))

                repo_trial_split = self.trainer.execute_trial(
                    pipeline=self.pipeline,
                    context=context,
                    trial_number=trial_number,
                    repo_trial=repo_trial,
                    validation_splits=validation_splits,
                    n_trial=self.n_trial
                )
        except self.error_types_to_raise as error:
            track = traceback.format_exc()
            repo_trial.set_failed(error)
            context.logger.critical(track)
            raise error
        except Exception:
            track = traceback.format_exc()
            repo_trial_split_number = 0 if repo_trial_split is None else repo_trial_split.split_number + 1
            context.logger.error('failed trial {}'.format(_get_trial_split_description(
                repo_trial=repo_trial,
                repo_trial_split_number=repo_trial_split_number,
                validation_splits=validation_splits,
                trial_number=trial_number,
                n_trial=self.n_trial
            )))
            context.logger.error(track)
        finally:
            repo_trial.update_final_trial_status()
            # Some heavy objects might have stayed in memory for a while during the execution of our trial;
            # It is best to do a full collection as that may free up some ram.
            gc.collect()

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> \
            ('BaseStep', DataContainer):
        raise NotImplementedError("AutoML does not implement method _fit_transform_data_container. Use method such as "
                                  "fit or handle_fit to train models and then use method such as get_best_model to "
                                  "retrieve the model you wish to use for transform")

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        raise NotImplementedError("AutoML does not implement method _transform_data_container. Use method such as "
                                  "get_best_model to retrieve the model you wish to use for transform")

    def get_best_model(self):
        """
        Get best model using the hyperparams repository.

        :return:
        """
        return self.hyperparams_repository.get_best_model()

    def _load_virgin_best_model(self) -> BaseStep:
        """
        Get the best model from all of the previous trials.

        :return: best model step
        """
        best_hyperparams = self.hyperparams_repository.get_best_hyperparams()
        p: Union[BaseCrossValidationWrapper, BaseStep] = copy.copy(self.pipeline)
        p = p.update_hyperparams(best_hyperparams)

        best_model = p.get_step()
        return copy.deepcopy(best_model)

    def _load_virgin_model(self, hyperparams: HyperparameterSamples) -> BaseStep:
        """
        Load virigin model with the given hyperparams.

        :return: best model step
        """
        return copy.deepcopy(self.pipeline).update_hyperparams(hyperparams)

    def get_children(self) -> List[BaseStep]:
        return [self.pipeline]


def _get_trial_split_description(
        repo_trial: Trial,
        repo_trial_split_number: int,
        validation_splits: List[Tuple[DataContainer, DataContainer]],
        trial_number: int,
        n_trial: int
):
    trial_split_description = '{}/{} split {}/{}\nhyperparams: {}'.format(
        trial_number + 1,
        n_trial,
        repo_trial_split_number + 1,
        len(validation_splits),
        json.dumps(repo_trial.hyperparams, sort_keys=True, indent=4)
    )
    return trial_split_description


class AutoMLContainer:
    """
    Data object for auto ml.

    .. seealso::
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.trial.Trials`,
        :class:`HyperparamsRepository`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(
            self,
            trials: 'Trials',
            hyperparameter_space: HyperparameterSpace,
            trial_number: int,
            main_scoring_metric_name: str
    ):
        self.trials = trials
        self.hyperparameter_space = hyperparameter_space
        self.trial_number = trial_number
        self.main_scoring_metric_name = main_scoring_metric_name


class RandomSearchHyperparameterSelectionStrategy(BaseHyperparameterSelectionStrategy):
    """
    AutoML Hyperparameter Optimizer that randomly samples the space of random variables.
    Please refer to :class:`AutoML` for a usage example.

    .. seealso::
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.trial.Trials`,
        :class:`HyperparamsRepository`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self):
        BaseHyperparameterSelectionStrategy.__init__(self)

    def find_next_best_hyperparams(self, auto_ml_container: 'AutoMLContainer') -> HyperparameterSamples:
        """
        Randomly sample the next hyperparams to try.

        :param auto_ml_container: trials data container
        :return: next best hyperparams
        """
        return auto_ml_container.hyperparameter_space.rvs()


class BaseValidationSplitter(ABC):
    def split_data_container(self, data_container: DataContainer, context: ExecutionContext) -> List[
        Tuple[DataContainer, DataContainer]]:
        """
        Wrap a validation split function with a split data container function.
        A validation split function takes two arguments:  data inputs, and expected outputs.

        :param data_container: data container to split
        :return: a function that returns the pairs of training, and validation data containers for each validation split.
        """
        train_data_inputs, train_expected_outputs, train_current_ids, validation_data_inputs, validation_expected_outputs, validation_current_ids = self.split(
            data_inputs=data_container.data_inputs,
            expected_outputs=data_container.expected_outputs,
            context=context
        )

        train_data_container = DataContainer(data_inputs=train_data_inputs,
                                             current_ids=train_current_ids,
                                             expected_outputs=train_expected_outputs)
        validation_data_container = DataContainer(data_inputs=validation_data_inputs,
                                                  current_ids=validation_current_ids,
                                                  expected_outputs=validation_expected_outputs)

        splits = []
        for (train_current_id, train_di, train_eo), (validation_current_id, validation_di, validation_eo) in zip(
                train_data_container, validation_data_container):
            train_data_container_split = DataContainer(
                summary_id=train_current_id,
                data_inputs=train_di,
                expected_outputs=train_eo
            )

            validation_data_container_split = DataContainer(
                summary_id=validation_current_id,
                data_inputs=validation_di,
                expected_outputs=validation_eo
            )

            splits.append((train_data_container_split, validation_data_container_split))

        return splits

    @abstractmethod
    def split(self, data_inputs, current_ids=None, expected_outputs=None, context: ExecutionContext = None) \
            -> Tuple[List, List, List, List, List, List]:
        """
        Train/Test split data inputs and expected outputs.

        :param data_inputs: data inputs
        :param current_ids: id associated with each data entry (optional)
        :param expected_outputs: expected outputs (optional)
        :param context: execution context (optional)
        :return: train_data_inputs, train_expected_outputs, train_current_ids, validation_data_inputs, validation_expected_outputs, validation_current_ids
        """
        pass


class KFoldCrossValidationSplitter(BaseValidationSplitter):
    """
    Create a function that splits data with K-Fold Cross-Validation resampling.

    .. code-block:: python

        # create a kfold cross validation splitter with 2 kfold
        kfold_cross_validation_split(0.20)


    :param k_fold: number of folds.
    :return:
    """

    def __init__(self, k_fold: int):
        BaseValidationSplitter.__init__(self)
        self.k_fold = k_fold

    def split(self, data_inputs, current_ids=None, expected_outputs=None, context: ExecutionContext = None) \
            -> Tuple[List, List, List, List, List, List]:
        data_inputs_train, data_inputs_val = kfold_cross_validation_split(
            data_inputs=data_inputs,
            k_fold=self.k_fold
        )

        if current_ids is not None:
            current_ids_train, current_ids_val = kfold_cross_validation_split(
                data_inputs=current_ids,
                k_fold=self.k_fold
            )
        else:
            current_ids_train, current_ids_val = [None] * len(data_inputs_train), [None] * len(data_inputs_val)

        if expected_outputs is not None:
            expected_outputs_train, expected_outputs_val = kfold_cross_validation_split(
                data_inputs=expected_outputs,
                k_fold=self.k_fold
            )
        else:
            expected_outputs_train, expected_outputs_val = [None] * len(data_inputs_train), [None] * len(
                data_inputs_val)

        return data_inputs_train, expected_outputs_train, current_ids_train, \
               data_inputs_val, expected_outputs_val, current_ids_val


def kfold_cross_validation_split(data_inputs, k_fold):
    splitted_train_data_inputs = []
    splitted_validation_inputs = []

    step = len(data_inputs) / float(k_fold)
    for i in range(k_fold):
        a = int(step * i)
        b = int(step * (i + 1))
        if b > len(data_inputs):
            b = len(data_inputs)

        validation = data_inputs[a:b]
        train = np.concatenate((data_inputs[:a], data_inputs[b:]), axis=0)

        splitted_validation_inputs.append(validation)
        splitted_train_data_inputs.append(train)

    return splitted_train_data_inputs, splitted_validation_inputs


class ValidationSplitter(BaseValidationSplitter):
    """
    Create a function that splits data into a training, and a validation set.

    .. code-block:: python

        # create a validation splitter function with 80% train, and 20% validation
        validation_splitter(0.20)


    :param test_size: test size in float
    :return:
    """

    def __init__(self, test_size: float):
        self.test_size = test_size

    def split(self, data_inputs, current_ids=None, expected_outputs=None, context: ExecutionContext = None) -> Tuple[
        List, List, List, List]:
        train_data_inputs, train_expected_outputs, train_current_ids, validation_data_inputs, validation_expected_outputs, validation_current_ids = validation_split(
            test_size=self.test_size,
            data_inputs=data_inputs,
            current_ids=current_ids,
            expected_outputs=expected_outputs
        )

        return [train_data_inputs], [train_expected_outputs], [train_current_ids], \
               [validation_data_inputs], [validation_expected_outputs], [validation_current_ids]


def validation_split(test_size: float, data_inputs, current_ids=None, expected_outputs=None) \
        -> Tuple[List, List, List, List, List, List]:
    """
    Split data inputs, and expected outputs into a training set, and a validation set.

    :param test_size: test size in float
    :param data_inputs: data inputs to split
    :param current_ids: ids associated with each data entry
    :param expected_outputs: expected outputs to split
    :return: train_data_inputs, train_expected_outputs, current_ids_train, validation_data_inputs, validation_expected_outputs, current_ids_val
    """
    validation_data_inputs = _validation_split(data_inputs, test_size)
    validation_expected_outputs, current_ids_val = None, None
    if expected_outputs is not None:
        validation_expected_outputs = _validation_split(expected_outputs, test_size)
    if current_ids is not None:
        current_ids_val = _validation_split(current_ids, test_size)

    train_data_inputs = _train_split(data_inputs, test_size)
    train_expected_outputs, current_ids_train = None, None
    if expected_outputs is not None:
        train_expected_outputs = _train_split(expected_outputs, test_size)
    if current_ids is not None:
        current_ids_train = _train_split(current_ids, test_size)

    return train_data_inputs, train_expected_outputs, current_ids_train, \
           validation_data_inputs, validation_expected_outputs, current_ids_val


def _train_split(data_inputs, test_size) -> List:
    """
    Split training set.

    :param data_inputs: data inputs to split
    :return: train_data_inputs
    """
    return data_inputs[0:_get_index_split(data_inputs, test_size)]


def _validation_split(data_inputs, test_size) -> List:
    """
    Split validation set.

    :param data_inputs: data inputs to split
    :return: validation_data_inputs
    """
    return data_inputs[_get_index_split(data_inputs, test_size):]


def _get_index_split(data_inputs, test_size):
    return math.floor(len(data_inputs) * (1 - test_size))
