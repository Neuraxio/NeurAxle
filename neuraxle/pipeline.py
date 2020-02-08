"""
Neuraxle's Pipeline Classes
====================================
This is the core of Neuraxle's pipelines. You can chain steps to call them one after an other.

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
import shutil
import warnings
from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Tuple, List

from neuraxle.base import BaseStep, TruncableSteps, NamedTupleList, ResumableStepMixin, NonFittableMixin, \
    ExecutionContext, ExecutionMode, NonTransformableMixin
from neuraxle.checkpoints import Checkpoint
from neuraxle.data_container import DataContainer, ListDataContainer

DEFAULT_CACHE_FOLDER = 'cache'


class BasePipeline(TruncableSteps, ABC):
    def __init__(self, steps: NamedTupleList):
        TruncableSteps.__init__(self, steps_as_tuple=steps)

    @abstractmethod
    def fit(self, data_inputs, expected_outputs=None) -> 'BasePipeline':
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data_inputs):
        raise NotImplementedError()

    @abstractmethod
    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BasePipeline', Any):
        raise NotImplementedError()


class Pipeline(BasePipeline):
    """
    Fits and transform steps
    """

    def __init__(self, steps: NamedTupleList, cache_folder=DEFAULT_CACHE_FOLDER):
        BasePipeline.__init__(self, steps=steps)
        self.cache_folder = cache_folder

    def transform(self, data_inputs: Any):
        """
        After loading the last checkpoint, transform each pipeline steps

        :param data_inputs: the data input to transform
        :return: transformed data inputs
        """
        data_container = DataContainer(data_inputs=data_inputs, current_ids=None)

        data_container = self.hash_data_container(data_container)
        context = ExecutionContext(root=self.cache_folder, execution_mode=ExecutionMode.TRANSFORM)
        context = context.push(self)
        data_container = self._transform_data_container(data_container, context)

        return data_container.data_inputs

    def transform_data_container(self, data_container: DataContainer):
        data_container = self.hash_data_container(data_container)
        context = ExecutionContext(root=self.cache_folder, execution_mode=ExecutionMode.TRANSFORM)
        context = context.push(self)
        data_container = self._transform_data_container(data_container, context)

        return data_container.data_inputs

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('Pipeline', Any):
        """
        After loading the last checkpoint, fit transform each pipeline steps

        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        data_container = DataContainer(data_inputs=data_inputs, current_ids=None, expected_outputs=expected_outputs)

        return self.fit_transform_data_container(data_container)

    def fit_transform_data_container(self, data_container):
        data_container = self.hash_data_container(data_container)
        context = ExecutionContext(root=self.cache_folder, execution_mode=ExecutionMode.FIT_TRANSFORM)
        context = context.push(self)
        new_self, data_container = self._fit_transform_data_container(data_container, context)

        return new_self, data_container.data_inputs

    def fit(self, data_inputs, expected_outputs=None) -> 'Pipeline':
        """
        After loading the last checkpoint, fit each pipeline steps

        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        data_container = DataContainer(data_inputs=data_inputs, current_ids=None, expected_outputs=expected_outputs)

        return self.fit_data_container(data_container)

    def fit_data_container(self, data_container):
        data_container = self.hash_data_container(data_container)
        context = ExecutionContext(self.cache_folder, ExecutionMode.FIT)
        context = context.push(self)
        new_self = self._fit_data_container(data_container, context)

        return new_self

    def inverse_transform(self, processed_outputs) -> Any:
        """
        After transforming all data inputs, and obtaining a prediction, we can inverse transform the processed outputs

        :param processed_outputs: the forward transformed data input
        :return: backward transformed processed outputs
        """
        for step_name, step in list(reversed(self.items())):
            processed_outputs = step.inverse_transform(processed_outputs)
        return processed_outputs

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> 'Pipeline':
        """
        After loading the last checkpoint, fit transform each pipeline steps,
        but only fit the last pipeline step.

        :param data_container: the data container to fit transform on
        :param context: execution context
        :return: tuple(pipeline, data_container)
        """
        steps_left_to_do, data_container = self._load_checkpoint(data_container, context)
        self.setup()

        index_last_step = len(steps_left_to_do) - 1

        new_steps_as_tuple: NamedTupleList = []

        for index, (step_name, step) in enumerate(steps_left_to_do):
            step.setup()

            if index != index_last_step:
                step, data_container = step.handle_fit_transform(data_container, context)
            else:
                step = step.handle_fit(data_container, context)

            new_steps_as_tuple.append((step_name, step))

        self.steps_as_tuple = self.steps_as_tuple[
                              :len(self.steps_as_tuple) - len(steps_left_to_do)] + new_steps_as_tuple

        return self

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('Pipeline', DataContainer):
        """
        After loading the last checkpoint, fit transform each pipeline steps

        :param data_container: the data container to fit transform on
        :param context: execution context
        :return: tuple(pipeline, data_container)
        """
        steps_left_to_do, data_container = self._load_checkpoint(data_container, context)
        self.setup()

        new_steps_as_tuple: NamedTupleList = []

        for step_name, step in steps_left_to_do:
            step.setup()
            step, data_container = step.handle_fit_transform(data_container, context)
            new_steps_as_tuple.append((step_name, step))

        self.steps_as_tuple = self.steps_as_tuple[:len(self.steps_as_tuple) - len(steps_left_to_do)] + \
                              new_steps_as_tuple

        return self, data_container

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        After loading the last checkpoint, transform each pipeline steps

        :param data_container: the data container to transform
        :return: transformed data container
        """
        steps_left_to_do, data_container = self._load_checkpoint(data_container, context)

        for step_name, step in steps_left_to_do:
            data_container = step.handle_transform(data_container, context)

        return data_container

    def _load_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> Tuple[NamedTupleList, DataContainer]:
        """
        Try loading a pipeline cache with the passed data container.
        If pipeline cache loading succeeds, find steps left to do,
        and load the latest data container.

        :param data_container: the data container to resume
        :param context: the execution context to resume
        :return: tuple(steps left to do, last checkpoint data container)
        """
        return self.steps_as_tuple, data_container

    def flush_all_cache(self):
        shutil.rmtree(self.cache_folder)


class ResumablePipeline(ResumableStepMixin, Pipeline):
    """
    Fits and transform steps after latest checkpoint
    """

    def _load_checkpoint(
            self,
            data_container: DataContainer,
            context: ExecutionContext
    ) -> Tuple[NamedTupleList, DataContainer]:
        """
        Try loading a pipeline cache with the passed data container.
        If pipeline cache loading succeeds, find steps left to do,
        and load the latest data container.

        :param data_container: the data container to resume
        :param context: the execution context to resume
        :return: tuple(steps left to do, last checkpoint data container)
        """
        new_starting_step_index, starting_step_data_container = \
            self._get_starting_step_info(data_container, context)

        loading_context = context.copy()
        loading_context.pop()
        loaded_pipeline = self.load(loading_context)

        if not self.are_steps_before_index_the_same(loaded_pipeline, new_starting_step_index):
            return self.steps_as_tuple, data_container

        self._assign_loaded_pipeline_into_self(loaded_pipeline)

        step = self[new_starting_step_index]
        if isinstance(step, Checkpoint):
            starting_step_data_container = step.read_checkpoint(starting_step_data_container, context)

        return self[new_starting_step_index:], starting_step_data_container

    def _assign_loaded_pipeline_into_self(self, loaded_self):
        self.steps_as_tuple = loaded_self.steps_as_tuple
        self._refresh_steps()
        self.hyperparams = loaded_self.hyperparams
        self.hyperparams_space = loaded_self.hyperparams_space

    def _get_starting_step_info(self, data_container: DataContainer, context: ExecutionContext) -> Tuple[int, DataContainer]:
        """
        Find the index of the latest step that can be resumed

        :param data_container: the data container to resume
        :return: index of the latest resumable step, data container at starting step
        """
        starting_step_data_container = copy(data_container)
        starting_step_context = copy(context)
        current_data_container = copy(data_container)
        index_latest_checkpoint = 0

        for index, (step_name, step) in enumerate(self.items()):
            if isinstance(step, ResumableStepMixin) and step.should_resume(current_data_container.copy(),
                                                                           starting_step_context):
                index_latest_checkpoint = index
                starting_step_data_container = copy(current_data_container)

            current_data_container = step.hash_data_container(current_data_container)

        return index_latest_checkpoint, starting_step_data_container

    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        """
        Return True if the pipeline has a saved checkpoint that it can resume from

        :param context: execution context
        :param data_container: the data container to resume
        :return: bool
        """
        context = context.push(self)
        for index, (step_name, step) in enumerate(reversed(self.items())):
            if isinstance(step, ResumableStepMixin) and step.should_resume(data_container, context):
                return True

        return False


class CustomPipelineMixin:
    def transform(self, data_inputs: Any):
        """
        :param data_inputs: the data input to transform
        :return: transformed data inputs
        """
        data_container = DataContainer(data_inputs=data_inputs, current_ids=None)

        self.hash_data_container(data_container)

        context = ExecutionContext(self.cache_folder, ExecutionMode.TRANSFORM)
        data_container = self.handle_transform(data_container, context)

        return data_container.data_inputs

    def fit(self, data_inputs, expected_outputs=None) -> 'Pipeline':
        """
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        self.setup()

        data_container = DataContainer(data_inputs=data_inputs, current_ids=None, expected_outputs=expected_outputs)
        current_ids = self.hash(data_container)
        data_container.set_current_ids(current_ids)

        context = ExecutionContext(self.cache_folder, ExecutionMode.FIT_TRANSFORM)
        new_self, data_container = self.handle_fit(data_container, context)

        return new_self

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('Pipeline', Any):
        """
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        self.setup()

        data_container = DataContainer(data_inputs=data_inputs, current_ids=None, expected_outputs=expected_outputs)

        data_container = self.hash_data_container(data_container)

        context = ExecutionContext(self.cache_folder, ExecutionMode.FIT_TRANSFORM)
        new_self, data_container = self.handle_fit_transform(data_container, context)

        return new_self, data_container.data_inputs


class MiniBatchSequentialPipeline(CustomPipelineMixin, Pipeline):
    """
    Mini Batch Sequential Pipeline class to create a pipeline processing data inputs in batch.
    Provide a default batch size :

    .. code-block:: python

        sub_pipelines = [SomeStep()]
        pipeline = MiniBatchSequentialPipeline(sub_pipelines, batch_size=32)


    Or manually add a :class`Barrier` step to the mini batch sequential pipeline :

    .. code-block:: python

        sub_pipelines = [SomeStep(), Joiner(32)]
        pipeline = MiniBatchSequentialPipeline(sub_pipelines)


    .. seealso::
        :class:`Pipeline`,
        :class:`Barrier`,
        :class:`Joiner`,
        :class:`DataContainer`,
        :class:`ExecutionContext`
    """

    def __init__(self, steps: NamedTupleList, batch_size=None):
        Pipeline.__init__(self, steps)
        CustomPipelineMixin.__init__(self)
        self.__validate_barriers_batch_size(batch_size)
        self.__patch_missing_barrier(batch_size)
        self.__patch_barriers_batch_size(batch_size)

    def __validate_barriers_batch_size(self, batch_size):
        if batch_size is not None:
            return

        for _, step in self:
            if isinstance(step, Barrier):
                if step.batch_size is None:
                    raise Exception(
                        'Invalid Joiner batch size {}[{}]. Please provide a default batch size to MiniBatchSequentialPipeline, or add a batch size to {}[{}].'.format(
                            self.name, step.name, self.name, step.name))

    def __patch_barriers_batch_size(self, batch_size):
        if batch_size is None:
            return

        for _, step in self:
            if isinstance(step, Barrier):
                if step.batch_size is not None:
                    warnings.warn(
                        'Replacing {}[{}].batch_size by {}.batch_size.'.format(self.name, step.name, self.name))
                step.batch_size = batch_size

    def __patch_missing_barrier(self, batch_size):
        has_barrier = False

        for _, step in self:
            if isinstance(step, Barrier):
                has_barrier = True

        if not has_barrier:
            self.steps_as_tuple.append(('Joiner', Joiner(batch_size)))

        self._refresh_steps()

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Transform all sub pipelines splitted by the Barrier steps.

        :param data_container: data container to transform.
        :param context: execution context
        :return: data container
        """
        sub_pipelines = self._create_sub_pipelines()

        for sub_pipeline in sub_pipelines:
            barrier = sub_pipeline[-1]
            data_container = barrier.join_transform(
                step=sub_pipeline,
                data_container=data_container,
                context=context
            )

            data_container = self.hash_data_container(data_container)

        return data_container

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext) -> 'MiniBatchSequentialPipeline':
        """
        Fit all sub pipelines splitted by the Barrier steps.

        :param data_container: data container to transform.
        :param context: execution context
        :return: data container
        """
        sub_pipelines = self._create_sub_pipelines()
        index_start = 0

        for sub_pipeline in sub_pipelines:
            sub_pipeline.setup()

            barrier = sub_pipeline[-1]
            sub_pipeline, data_container = barrier.join_fit_transform(
                step=sub_pipeline,
                data_container=data_container,
                context=context
            )
            current_ids = self.hash(data_container)
            data_container.set_current_ids(current_ids)

            new_self = self[:index_start] + sub_pipeline
            if index_start + len(sub_pipeline) < len(self):
                new_self += self[index_start + len(sub_pipeline):]

            self.steps_as_tuple = new_self.steps_as_tuple
            index_start += len(sub_pipeline)

        return self

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> Tuple['MiniBatchSequentialPipeline', DataContainer]:
        """
        Transform all sub pipelines splitted by the Barrier steps.

        :param data_container: data container to transform.
        :param context: execution context
        :return: data container
        """
        sub_pipelines = self._create_sub_pipelines()
        index_start = 0

        for sub_pipeline in sub_pipelines:
            sub_pipeline.setup()

            barrier = sub_pipeline[-1]
            sub_pipeline, data_container = barrier.join_fit_transform(
                step=sub_pipeline,
                data_container=data_container,
                context=context
            )

            data_container = self.hash_data_container(data_container)

            new_self = self[:index_start] + sub_pipeline
            if index_start + len(sub_pipeline) < len(self):
                new_self += self[index_start + len(sub_pipeline):]

            self.steps_as_tuple = new_self.steps_as_tuple
            index_start += len(sub_pipeline)

        return self, data_container

    def _create_sub_pipelines(self) -> List['MiniBatchSequentialPipeline']:
        """
        Create sub pipelines by splitting the steps by the join type name.

        :return: list of sub pipelines
        """
        sub_pipelines: List[MiniBatchSequentialPipeline] = self.split(Barrier)
        for sub_pipeline in sub_pipelines:
            if not sub_pipeline.ends_with(Barrier):
                raise Exception(
                    'At least one Barrier step needs to be at the end of a streaming pipeline.'
                )

        return sub_pipelines


class Barrier(NonFittableMixin, NonTransformableMixin, BaseStep, ABC):
    """
    A Barrier step to be used in a minibatch sequential pipeline. It forces all the
    data inputs to get to the barrier in a sub pipeline before going through to the next sub-pipeline.

    ```
    p = MiniBatchSequentialPipeline([
        SomeStep(),
        SomeStep(),
        Barrier(), # must be a concrete Barrier ex: Joiner()
        SomeStep(),
        SomeStep(),
        Barrier(), # must be a concrete Barrier ex: Joiner()
    ], batch_size=10)
    ```
    """

    @abstractmethod
    def join_transform(self, step: TruncableSteps, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Execute the given pipeline :func:`~neuraxle.pipeline.Pipeline.transform` with the given data container, and execution context.

        :param step: truncable steps to execute
        :type step: TruncableSteps
        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: transformed data container
        :rtype: DataContainer
        """
        raise NotImplementedError()

    @abstractmethod
    def join_fit_transform(self, step: Pipeline, data_container: DataContainer, context: ExecutionContext) -> Tuple['Any', DataContainer]:
        """
        Execute the given pipeline :func:`~neuraxle.pipeline.Pipeline.fit_transform` with the given data container, and execution context.

        :param step: truncable steps to execute
        :type step: Pipeline
        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: (fitted step, transformed data container)
        :rtype: Tuple['Any', DataContainer]
        """
        raise NotImplementedError()


class Joiner(Barrier):
    """
    A Special Barrier step that joins the transformed mini batches together with list.extend method.
    """

    def __init__(self, batch_size):
        Barrier.__init__(self)
        self.batch_size = batch_size

    def join_transform(self, step: Pipeline, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Concatenate the pipeline transform output of each batch of self.batch_size together.

        :param step: pipeline to transform on
        :type step: Pipeline
        :param data_container: data container to transform
        :type data_container: DataContainer
        :param context: execution context
        :return: transformed data container
        :rtype: DataContainer
        """
        context = context.push(step)

        data_container_batches = data_container.convolved_1d(
            stride=self.batch_size,
            kernel_size=self.batch_size
        )

        output_data_container = ListDataContainer.empty()
        for data_container_batch in data_container_batches:
            output_data_container.concat(
                step._transform_data_container(data_container_batch, context)
            )

        return output_data_container

    def join_fit_transform(self, step: Pipeline, data_container: DataContainer, context: ExecutionContext) -> Tuple['Any', DataContainer]:
        """
        Concatenate the pipeline fit transform output of each batch of self.batch_size together.

        :param step: pipeline to fit transform on
        :type step: Pipeline
        :param data_container: data container to fit transform on
        :type data_container: DataContainer
        :param context: execution context
        :return: fitted self, transformed data inputs
        :rtype: Tuple[Any, DataContainer]
        """
        context = context.push(step)

        data_container_batches = data_container.convolved_1d(
            stride=self.batch_size,
            kernel_size=self.batch_size
        )

        output_data_container = ListDataContainer.empty()
        for data_container_batch in data_container_batches:
            step, data_container_batch = step._fit_transform_data_container(data_container_batch, context)
            output_data_container.concat(
                data_container_batch
            )

        return step, output_data_container
