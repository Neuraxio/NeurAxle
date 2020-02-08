import numpy as np

from neuraxle.base import NonFittableMixin, BaseStep
from neuraxle.distributed.streaming import SequentialQueuedPipeline, ParallelQueuedPipeline

EXPECTED_OUTPUTS = np.array(range(100)) * 2 * 2 * 2 * 2
EXPECTED_OUTPUTS_PARALLEL = np.array((np.array(range(100)) * 2).tolist() * 4)


class MultiplyBy(NonFittableMixin, BaseStep):
    def __init__(self, multiply_by=1):
        NonFittableMixin.__init__(self)
        BaseStep.__init__(self)
        self.multiply_by = multiply_by

    def transform(self, data_inputs):
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs * self.multiply_by

    def inverse_transform(self, data_inputs):
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs / self.multiply_by


def test_queued_pipeline_with_step_name_n_worker_max_size():
    # Given
    p = SequentialQueuedPipeline([
        ('1', 1, 5, MultiplyBy(2)),
        ('2', 1, 5, MultiplyBy(2)),
        ('3', 1, 5, MultiplyBy(2)),
        ('4', 1, 5, MultiplyBy(2))
    ], batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_size():
    # Given
    p = SequentialQueuedPipeline([
        ('1', 1, MultiplyBy(2)),
        ('2', 1, MultiplyBy(2)),
        ('3', 1, MultiplyBy(2)),
        ('4', 1, MultiplyBy(2))
    ], max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_with_default_n_workers_and_default_max_size():
    # Given
    p = SequentialQueuedPipeline([
        ('1', MultiplyBy(2)),
        ('2', MultiplyBy(2)),
        ('3', MultiplyBy(2)),
        ('4', MultiplyBy(2))
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_parallel_queued_pipeline_with_step_name_n_worker_max_size():
    # Given
    p = ParallelQueuedPipeline([
        ('1', 1, 5, MultiplyBy(2)),
        ('2', 1, 5, MultiplyBy(2)),
        ('3', 1, 5, MultiplyBy(2)),
        ('4', 1, 5, MultiplyBy(2))
    ], batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_with_step_name_n_worker_additional_arguments_max_size():
    n_workers = 4
    worker_arguments = [('multiply_by', 2) for _ in range(n_workers)]
    p = ParallelQueuedPipeline([
        ('1', n_workers, worker_arguments, 5, MultiplyBy()),
    ], batch_size=10)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    expected = np.array(list(range(0, 200, 2)))
    assert np.array_equal(outputs, expected)


def test_parallel_queued_pipeline_with_step_name_n_worker_additional_arguments():
    n_workers = 4
    worker_arguments = [('multiply_by', 2) for _ in range(n_workers)]
    p = ParallelQueuedPipeline([
        ('1', n_workers, worker_arguments, MultiplyBy()),
    ], batch_size=10, max_size=5)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    expected = np.array(list(range(0, 200, 2)))
    assert np.array_equal(outputs, expected)


def test_parallel_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_size():
    # Given
    p = ParallelQueuedPipeline([
        ('1', 1, MultiplyBy(2)),
        ('2', 1, MultiplyBy(2)),
        ('3', 1, MultiplyBy(2)),
        ('4', 1, MultiplyBy(2)),
    ], max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_with_step_name_n_worker_with_default_n_workers_and_default_max_size():
    # Given
    p = ParallelQueuedPipeline([
        ('1', MultiplyBy(2)),
        ('2', MultiplyBy(2)),
        ('3', MultiplyBy(2)),
        ('4', MultiplyBy(2)),
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_step_name_n_worker_with_default_n_workers_and_default_max_size():
    # Given
    p = ParallelQueuedPipeline([
        ('1', MultiplyBy(2)),
        ('2', MultiplyBy(2)),
        ('3', MultiplyBy(2)),
        ('4', MultiplyBy(2)),
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)
