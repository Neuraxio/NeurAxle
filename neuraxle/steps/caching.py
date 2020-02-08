"""
Pipeline Steps For Caching
=====================================

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
import hashlib
import os
import pickle
import shutil
from abc import abstractmethod, ABC
from typing import Iterable, Any

from neuraxle.base import MetaStepMixin, BaseStep, NonFittableMixin, NonTransformableMixin, \
    ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.pipeline import DEFAULT_CACHE_FOLDER
from neuraxle.steps.misc import VALUE_CACHING


class ValueCachingWrapper(MetaStepMixin, BaseStep):
    """
    Value caching wrapper wraps a step to cache the values.
    """

    def __init__(
            self,
            wrapped: BaseStep,
            cache_folder: str = DEFAULT_CACHE_FOLDER,
            value_hasher: 'BaseValueHasher' = None,
    ):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        self.value_hasher = value_hasher

        if self.value_hasher is None:
            self.value_hasher = Md5Hasher()

        self.cache_folder = cache_folder

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Fit transform data container.

        :param context: execution context
        :param data_container: the data container to transform
        :type data_container: neuraxle.data_container.DataContainer

        :return: tuple(fitted pipeline, data_container)
        """
        self.create_checkpoint_path(context.get_path())
        self.flush_cache()

        self.wrapped = self.wrapped.fit(data_container.data_inputs, data_container.expected_outputs)
        outputs = self._transform_with_cache(data_container)
        data_container.set_data_inputs(outputs)

        return self, data_container

    def _transform_data_container(self, data_container, context):
        """
        Transform data container.

        :param context: execution context
        :param data_container: the data container to transform
        :type data_container: neuraxle.data_container.DataContainer

        :return: transformed data container
        """
        self.create_checkpoint_path(context.get_path())
        outputs = self._transform_with_cache(data_container)
        data_container.set_data_inputs(outputs)

        return data_container

    def _hash_value(self, data_input):
        return self.value_hasher.hash(data_input)

    def _transform_with_cache(self, data_container: DataContainer) -> Iterable:
        """
        Transform data container using value caching.

        :param data_container: the data container to transform
        :type data_container: neuraxle.data_container.DataContainer

        :return: iterable
        """
        outputs = []
        for current_id, data_input, expected_output in data_container:
            if self.contains_cache_for(data_input):
                outputs.extend(self.read_cache(data_input))
            else:
                output = self.wrapped.transform([data_input])
                self.write_cache(data_input, output)
                outputs.extend(output)
        return outputs

    @abstractmethod
    def create_checkpoint_path(self, step_path: str) -> str:
        """
        Create checkpoint path.

        :param step_path: step path inside pipeline ex: ``Pipeline/step_name/`` 
        :type step_path: str

        :return: checkpoint path
        """
        raise NotImplementedError()

    @abstractmethod
    def flush_cache(self):
        """
        Flush all cached values
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def read_cache(self, data_input) -> Any:
        """
        Read cache for a given data input.

        :param data_input: data input to get cache for
        :type data_input: Any

        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def write_cache(self, data_input, output):
        """
        Write cache for a given data input and output.

        :param data_input: data input to write cache for
        :type data_input: Any

        :param output: output to write cache for
        :type output: Any

        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def contains_cache_for(self, data_input) -> bool:
        """
        Returns true if the data input transform output is cached.

        :param data_input: to get cache from
        :return: boolean to indicate if a cache is present for the given data input
        """
        raise NotImplementedError()

    @abstractmethod
    def get_cache_path_for(self, data_input) -> str:
        """
        Get the cache path for the given data input.

        :param data_input: data input to get cache path for
        :return: str for cache path
        """
        raise NotImplementedError()


class PickleValueCachingWrapper(ValueCachingWrapper):
    """
    Value Caching Wrapper class that caches the wrapped step transformed data inputs using python ``pickle`` library.
    """

    def transform(self, data_inputs):
        pass

    def create_checkpoint_path(self, step_path: str) -> str:
        self.checkpoint_path = os.path.join(self.cache_folder, step_path, VALUE_CACHING)

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        return self.checkpoint_path

    def flush_cache(self):
        shutil.rmtree(self.checkpoint_path)
        os.mkdir(self.checkpoint_path)

    def read_cache(self, data_input):
        with open(self.get_cache_path_for(data_input), 'rb') as file_:
            return pickle.load(file_)

    def write_cache(self, data_input, output):
        with open(self.get_cache_path_for(data_input), 'wb') as file_:
            return pickle.dump(output, file_)

    def contains_cache_for(self, data_input) -> bool:
        return os.path.exists(self.get_cache_path_for(data_input))

    def get_cache_path_for(self, data_input):
        hash_value = self._hash_value(data_input)
        return os.path.join(self.checkpoint_path, '{0}.pickle'.format(hash_value))


class BaseValueHasher(ABC):
    @abstractmethod
    def hash(self, data_input):
        raise NotImplementedError()


class Md5Hasher(BaseValueHasher):
    def hash(self, data_input):
        m = hashlib.md5()
        m.update(str.encode(str(data_input)))

        return m.hexdigest()
