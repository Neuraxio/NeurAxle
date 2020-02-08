"""
Neuraxle steps for parallel processing
================================================

Neuraxle Steps for parallel processing

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

"""
import math
import os
import tempfile
import time
import warnings
from typing import List, Iterable, Tuple

import fs
import numpy as np
from fs.memoryfs import MemoryFS
from joblib import dump, Parallel, delayed, load

from neuraxle.base import BaseStep, MetaStepMixin, NonFittableMixin, ExecutionContext, Identity, BaseSaver, \
    DEFAULT_CACHE_FOLDER, ExecutionMode, FullDumpLoader
from neuraxle.data_container import DataContainer


class MemoryFSJoblibSaver(BaseSaver):
    """
    Saver that saves steps in a volatile in-memory file system.
    This saver is used by :class:`SaverParallelTransform` to avoid extra memory overhead.

    Using Python's Filesystem abstraction layer : `MemoryFS <http://cnn.com>`_

    .. seealso::
        :class:`BaseSaver`,
        :func:`~neuraxle.base.BaseStep.load`,
        :func:`~neuraxle.base.BaseStep.save`,
        :class:`SaverParallelTransform`,
        :class:`ExecutionContext`
    """

    def __init__(self, memory_file_system: MemoryFS):
        self.memory_file_system = memory_file_system

    def save_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Saved step stripped out of things that would make it unserializable.

        :param step: stripped step to save
        :type step: BaseStep
        :param context: execution context to save from
        :type context: ExecutionContext
        :return:
        """
        context.mkdir()
        step_path = self._create_step_path(context, step)

        self.memory_file_system.touch(step_path)

        dump(step, step_path)

        with self.memory_file_system.openbin(step_path, mode='w') as file:
            dump(step, file)

        return step

    def can_load(self, step: 'BaseStep', context: 'ExecutionContext'):
        """
        Returns true if the given step has been saved with the given execution context.

        :param step: step that might have been saved
        :type step: BaseStep
        :param context: execution context
        :type context: ExecutionContext
        :return: if we can load the step with the given context
        :rtype: bool
        """
        return os.path.exists(
            self._create_step_path(context, step)
        )
        # return self.memory_file_system.exists(self._create_step_path(context, step))

    def load_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Load stripped step.

        :param step: stripped step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return:
        """
        step_path = self._create_step_path(context, step)

        return load(step_path)
        # return load(self.memory_file_system.openbin(step_path))

    def _create_step_path(self, context, step):
        return os.path.join(context.get_path(), '{0}.joblib'.format(step.name))


class MemoryFSExecutionContext(ExecutionContext):
    """
    The execution context is created in a volatile in-memory file system.
    It uses :class:`MemoryFSJoblibSaver` as the default stripped saver.

    Please refer to :class:`ExecutionContext` for more info.

    .. seealso::
        :class:`ExecutionContext`,
        :class:`MemoryFSJoblibSaver`
    """

    def __init__(
            self,
            memory_file_system: MemoryFS,
            root: str = DEFAULT_CACHE_FOLDER,
            execution_mode: ExecutionMode = None,
            parents=None
    ):

        ExecutionContext.__init__(
            self,
            root=root,
            execution_mode=execution_mode,
            stripped_saver=MemoryFSJoblibSaver(memory_file_system),
            parents=parents
        )
        self.memory_file_system = memory_file_system

    def mkdir(self):
        """
        Creates the last parent step directory in the memory file system.

        :return:
        """
        super().mkdir()
        path = self.get_path()
        parts = path.split(os.sep)

        dir_to_create = ''
        for i in range(len(parts)):
            dir_to_create += parts[i] + os.sep
            if not self.memory_file_system.exists(dir_to_create):
                self.memory_file_system.makedir(dir_to_create)

    def push(self, step: 'BaseStep'):
        """
        Pushes a step in the parents of the execution context.

        :param step: step to add to the execution context
        :type step: BaseStep
        :return: self
        :rtype: ExecutionContext
        """
        return MemoryFSExecutionContext(
            memory_file_system=self.memory_file_system,
            root=self.root,
            execution_mode=self.execution_mode,
            parents=self.parents + [step],
        )

    def to_identity(self) -> 'MemoryFSExecutionContext':
        """
        Create a fake execution context containing only identity steps.
        Create the parents by using the path of the current execution context.

        :return: fake identity execution context
        :rtype: MemoryFSExecutionContext

        .. seealso::
            :class:`FullDumpLoader`,
            :class:`Identity`
        """
        step_names = self.get_path(False).split(os.sep)

        parents = [
            Identity(name=name)
            for name in step_names
        ]

        return MemoryFSExecutionContext(
            memory_file_system=self.memory_file_system,
            root=self.root,
            execution_mode=self.execution_mode,
            parents=parents
        )


class SaverParallelTransform(NonFittableMixin, MetaStepMixin, BaseStep):
    """
    Use savers to parallelize steps transformations to avoid python limitations when importing external librairies.
    Dispatching technique class to abstract the workers.

    .. seealso::
        :func:`~NonFittableMixin`,
        :func:`~MetaStepMixin`,
        :class:`BaseStep`
    """

    def __init__(self, wrapped: BaseStep, mount_path=None, n_jobs: int = None, batch_size=None):
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)

        if mount_path is None:
            mount_path = 'cache'

        if not os.path.ismount(mount_path):
            warnings.warn('SaverParallelTransform.mount_path not mounted.', RuntimeWarning, stacklevel=2)
        else:
            print('mounted !')
        self.mount_path = mount_path
        self.n_jobs = n_jobs
        self.batch_size = batch_size

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        """
        Parallelize transform with a volatile in-memory file system.
        Save a full dump of the pipeline in memory.
        Send batches of the data container to `joblib.Parallel <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_
        """
        new_context = context.copy()
        new_context.root = self.mount_path
        a = time.time()
        context = self._save_shared_memory_execution_context(new_context)
        b = time.time()
        print('WRITING')
        print(b-a)
        print('\n')

        batch_size = self._get_batch_size(data_container)
        data_container_batches = data_container.convolved_1d(stride=batch_size, kernel_size=batch_size)
        step_path = context.get_path(with_root=False)

        a = time.time()
        outputs_data_containers = Parallel(
                n_jobs=self.n_jobs,
                batch_size=self.batch_size,
                backend='multiprocessing',
        )(delayed(receive)(self.mount_path, step_path, data_container_batch) for data_container_batch in data_container_batches)
        b = time.time()

        print('READING')
        print(b-a)
        print('\n')

        data_inputs, expected_outputs = self._join_output_data_containers(outputs_data_containers)
        data_container.set_data_inputs(np.array(data_inputs))
        data_container.set_expected_outputs(np.array(expected_outputs))

        return data_container

    def _join_output_data_containers(self, outputs_data_containers: List[DataContainer]) -> Tuple[Iterable, Iterable]:
        """
        Join output data containers together.

        :return: joined_data_inputs, joined_expected_outputs
        :rtype: Tuple[Iterable, Iterable]
        """
        data_inputs = []
        expected_outputs = []

        for output_data_container in outputs_data_containers:
            data_inputs.extend(output_data_container.data_inputs)
            expected_outputs.extend(output_data_container.expected_outputs)

        return data_inputs, expected_outputs

    def _get_batch_size(self, data_container: DataContainer) -> int:
        """
        Get batch size.

        :param data_container: data container
        :type data_container: DataContainer
        :return: batch_size
        :rtype: int
        """
        if self.batch_size is None:
            batch_size = math.ceil(len(data_container) / self.n_jobs)
        else:
            batch_size = self.batch_size
        return batch_size

    def _save_shared_memory_execution_context(self, context: ExecutionContext):
        """
        Save a full dump of the execution context in shared memory.

        :param context: execution context
        :type context: ExecutionContext
        :param memory_file_system: memory file system
        :type memory_file_system: MemoryFS
        :return: batch_size
        :rtype: int
        """
        identity = context.to_identity()
        context.save_last()
        return identity

    def transform(self, data_inputs):
        raise Exception(
            'Transform method not supported by SharedMemoryDispatcher. Please use this step inside a pipeline'.format(
                repr(self)))


def receive(cache_folder: str, step_path: str, data_container: DataContainer):
    """
    Save a full dump of the execution context in shared memory.

    :param cache_folder: cache folder
    :type cache_folder: List[str]
    :param step_path: step names
    :type step_path: List[str]
    :type data_container: DataContainer
    :param data_container: data container
    :return: transformed data container
    :rtype: DataContainer
    """
    context = ExecutionContext(cache_folder)
    step: SaverParallelTransform = context.load(step_path)

    return step.wrapped.handle_transform(data_container, context)
