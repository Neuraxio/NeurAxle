import numpy as np
import pytest
from joblib import Parallel, delayed
from neuraxle.metaopt.trial import TRIAL_STATUS, Trials
from sklearn.metrics import mean_squared_error

from neuraxle.hyperparams.distributions import Uniform, LogNormal, Normal, Choice, Quantized, LogUniform
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import InMemoryHyperparamsRepository, AutoML, ValidationSplitter, \
    RandomSearchHyperparameterSelectionStrategy, BaseHyperparameterSelectionStrategy
from neuraxle.metaopt.callbacks import MetricCallback, ScoringCallback
from neuraxle.metaopt.tpe import TreeParzenEstimatorHyperparameterSelectionStrategy
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitTransformCallbackStep
from neuraxle.steps.numpy import AddN
import os


@pytest.mark.parametrize("expected_output_mult, pipeline", [
    (3.5, Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Choice(choice_list=[0, 1.5, 2, 3.5, 4, 5, 6]),
        })),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Choice(choice_list=[0, 1.5, 2, 3.5, 4, 5, 6]),
        }))
    ])),
    (3.5, Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Quantized(hd=Uniform(0, 10)),
        })),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Quantized(hd=Uniform(0, 10)),
        }))
    ])),
    (3.5, Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': LogUniform(min_included=1.0, max_included=5.0),
        })),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': LogUniform(min_included=1.0, max_included=5.0),
        }))
    ])),
    (3.5, Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': LogNormal(log2_space_mean=1.0, log2_space_std=0.5),
        })),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': LogNormal(log2_space_mean=1.0, log2_space_std=0.5),
        }))
    ])),
    (3.5, Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Normal(mean=3.0, std=2.0, hard_clip_min=0, hard_clip_max=6),
        })),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Normal(mean=3.0, std=2.0, hard_clip_min=0, hard_clip_max=6),
        })),
    ])),
    (3.5, Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Uniform(0, 6),
        })),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Uniform(0, 6),
        }))
    ]))
])
def test_tpe(expected_output_mult, pipeline, tmpdir):
    # set random seed
    np.random.seed(42)

    # Given
    tpe_scores = Parallel(n_jobs=-2)(
        delayed(_test_trial_scores)(
            expected_output_mult,
            pipeline,
            TreeParzenEstimatorHyperparameterSelectionStrategy(
                number_of_initial_random_step=5,
                quantile_threshold=0.3,
                number_good_trials_max_cap=25,
                number_possible_hyperparams_candidates=100,
                prior_weight=0.,
                use_linear_forgetting_weights=False,
                number_recent_trial_at_full_weights=25
            ),
            os.path.join(tmpdir, 'tpe', str(i))
        ) for i in range(4)
    )

    random_scores = Parallel(n_jobs=-2)(
        delayed(_test_trial_scores)(
            expected_output_mult,
            pipeline,
            RandomSearchHyperparameterSelectionStrategy(),
            os.path.join(tmpdir, 'random', str(i))
        )
        for i in range(4)
    )

    mean_tpe_score = np.array(tpe_scores).flatten().mean(axis=-1)
    mean_random_score = np.array(random_scores).flatten().mean(axis=-1)

    assert mean_tpe_score < mean_random_score



def _test_trial_scores(
    expected_output_mult,
    pipeline,
    hyperparams_optimizer: BaseHyperparameterSelectionStrategy,
    tmpdir: str
):
    hp_repository: InMemoryHyperparamsRepository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 1
    n_trials = 10
    auto_ml: AutoML = AutoML(
        pipeline=pipeline,
        hyperparams_optimizer=hyperparams_optimizer,
        validation_splitter=ValidationSplitter(0.5),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False),
        ],
        n_trials=n_trials,
        refit_trial=True,
        epochs=n_epochs,
        hyperparams_repository=hp_repository,
        continue_loop_on_error=False
    )

    # When
    data_inputs = np.array([0, 0])
    expected_outputs = expected_output_mult * np.ones_like(data_inputs)
    auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    trials: Trials = hp_repository.load_all_trials(status=TRIAL_STATUS.SUCCESS)
    validation_scores = [t.get_validation_score() for t in trials]
    return validation_scores

