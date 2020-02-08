import os

import numpy as np

from neuraxle.base import Identity, ExecutionContext, JoblibStepSaver, FullDumpLoader
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitTransformCallbackStep, TapeCallbackFunction
from neuraxle.steps.output_handlers import OutputTransformerWrapper

PIPELINE_NAME = 'saved_pipeline'

DATA_INPUTS = np.array(range(10))
EXPECTED_OUTPUTS = np.array(range(10, 20))


def test_load_full_dump_from_pipeline_name(tmpdir):
    # Given
    tape_fit_callback_function = TapeCallbackFunction()
    tape_transform_callback_function = TapeCallbackFunction()
    pipeline = Pipeline([
        ('step_a', Identity()),
        ('step_b', OutputTransformerWrapper(
            FitTransformCallbackStep(tape_fit_callback_function, tape_transform_callback_function)
        ))
    ], cache_folder=tmpdir).set_name(PIPELINE_NAME)

    # When
    pipeline, outputs = pipeline.fit_transform(DATA_INPUTS, EXPECTED_OUTPUTS)
    pipeline.save(ExecutionContext(tmpdir), full_dump=True)

    # Then
    loaded_pipeline = ExecutionContext(tmpdir).load(PIPELINE_NAME)

    assert isinstance(loaded_pipeline, Pipeline)
    assert isinstance(loaded_pipeline['step_a'], Identity)
    assert isinstance(loaded_pipeline['step_b'], OutputTransformerWrapper)

    loaded_step_b_wrapped_step = loaded_pipeline['step_b'].wrapped
    assert np.array_equal(loaded_step_b_wrapped_step.transform_callback_function.data[0], EXPECTED_OUTPUTS)
    assert np.array_equal(loaded_step_b_wrapped_step.fit_callback_function.data[0][0], EXPECTED_OUTPUTS)
    assert np.array_equal(loaded_step_b_wrapped_step.fit_callback_function.data[0][1], [None] * len(EXPECTED_OUTPUTS))


def test_load_full_dump_from_path(tmpdir):
    # Given
    tape_fit_callback_function = TapeCallbackFunction()
    tape_transform_callback_function = TapeCallbackFunction()
    pipeline = Pipeline([
        ('step_a', Identity()),
        ('step_b', OutputTransformerWrapper(
            FitTransformCallbackStep(tape_fit_callback_function, tape_transform_callback_function)
        ))
    ], cache_folder=tmpdir).set_name(PIPELINE_NAME)

    # When
    pipeline, outputs = pipeline.fit_transform(DATA_INPUTS, EXPECTED_OUTPUTS)
    pipeline.save(ExecutionContext(tmpdir), full_dump=True)

    # Then
    loaded_pipeline = ExecutionContext(tmpdir).load(os.path.join(PIPELINE_NAME, 'step_b'))

    assert isinstance(loaded_pipeline, OutputTransformerWrapper)
    loaded_step_b_wrapped_step = loaded_pipeline.wrapped
    assert np.array_equal(loaded_step_b_wrapped_step.transform_callback_function.data[0], EXPECTED_OUTPUTS)
    assert np.array_equal(loaded_step_b_wrapped_step.fit_callback_function.data[0][0], EXPECTED_OUTPUTS)
    assert np.array_equal(loaded_step_b_wrapped_step.fit_callback_function.data[0][1], [None] * len(EXPECTED_OUTPUTS))
