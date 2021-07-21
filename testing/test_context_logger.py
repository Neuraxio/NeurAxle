import logging
import os
import shutil

import numpy as np
from sklearn.metrics import mean_squared_error

from neuraxle.base import BaseStep, ExecutionContext, HandleOnlyMixin
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.distributions import FixedHyperparameter
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import AutoML, InMemoryHyperparamsRepository, RandomSearchHyperparameterSelectionStrategy, \
    ValidationSplitter, HyperparamsJSONRepository
from neuraxle.metaopt.callbacks import ScoringCallback
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import MultiplyByN, NumpyReshape


logging_call_counter = 0

class LoggingStep(HandleOnlyMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        HandleOnlyMixin.__init__(self)


    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> BaseStep:
        global logging_call_counter
        context.logger.info(f"fit call - logging call # {logging_call_counter}")
        logging_call_counter += 1
        return self

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        global logging_call_counter
        context.logger.info(f"transform call - logging call # {logging_call_counter}")
        logging_call_counter += 1
        return data_container


def test_logger():
    file_path = "test.log"

    if os.path.exists(file_path):
        os.remove(file_path)

    #Given
    logger = logging.getLogger('test')
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel('DEBUG')
    logger.addHandler(file_handler)
    logger.setLevel('DEBUG')
    context = ExecutionContext(logger=logger)
    pipeline = Pipeline([
            MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
                'multiply_by': FixedHyperparameter(2)
            })),
            NumpyReshape(new_shape=(-1, 1)),
            LoggingStep()
        ])

    # When
    data_container = DataContainer(
        data_inputs=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    )
    pipeline.handle_fit(data_container, context)

    # Then
    assert os.path.exists(file_path)
    with open(file_path) as f:
        l = f.read()

    # Teardown
    os.remove(file_path)

class TestTrialLogger:
    def test_logger_automl(self, tmpdir):
        # Given
        context = ExecutionContext()
        self.tmpdir = str(tmpdir)
        hp_repository = HyperparamsJSONRepository(cache_folder=self.tmpdir)
        n_epochs = 2
        n_trials = 4
        auto_ml = AutoML(
            pipeline=Pipeline([
                MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
                    'multiply_by': FixedHyperparameter(2)
                })),
                NumpyReshape(new_shape=(-1, 1)),
                LoggingStep()
            ]),
            hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
            validation_splitter=ValidationSplitter(0.20),
            scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
            n_trials=n_trials,
            refit_trial=True,
            epochs=n_epochs,
            hyperparams_repository=hp_repository,
            continue_loop_on_error=False
        )

        # When
        data_container = DataContainer(
            data_inputs=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            expected_outputs=np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        )
        auto_ml.handle_fit(data_container, context)

        # Then
        file_paths = [os.path.join(hp_repository.cache_folder, f"trial_{i}.log") for i in range(n_trials)]

        for f in file_paths:
            assert os.path.exists(f)

        for f in file_paths:
            with open(f, 'r') as f:
                l = f.readlines()
                # That not a great way of testing... but at least it raises a flag when something changes in the logging process
#                assert len(l) == 36


    def teardown(self):
        shutil.rmtree(self.tmpdir)