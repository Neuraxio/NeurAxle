"""
Neuraxle's Base Classes
====================================
This is the core of Neuraxle. Most pipeline steps derive (inherit) from those classes. They are worth noticing.

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
import inspect
import os
import pprint
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy
from enum import Enum
from typing import Tuple, List, Union, Any, Iterable, KeysView, ItemsView, ValuesView, Callable, Optional

from joblib import dump, load
from sklearn.base import BaseEstimator

from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples

DEFAULT_CACHE_FOLDER = os.path.join(os.getcwd(), 'cache')


class BaseHasher(ABC):
    """
    Base class to hash hyperparamters, and data input ids together.
    The :class:`DataContainer` class uses the hashed values for its current ids.
    :class:`BaseStep` uses many :class:`BaseHasher` objects
    to hash hyperparameters, and data inputs ids together after each transform.

    .. seealso::
        :class:`DataContainer`

    """

    @abstractmethod
    def single_hash(self, current_id: str, hyperparameters: HyperparameterSamples) -> List[str]:
        """
        Hash summary id, and hyperparameters together.

        :param current_id: current hashed id
        :param hyperparameters: step hyperparameters to hash with current ids
        :type hyperparameters: HyperparameterSamples
        :return: the new hashed current id
        :rtype: str
        """
        raise NotImplementedError()

    @abstractmethod
    def hash(self, current_ids: List[str], hyperparameters: HyperparameterSamples, data_inputs: Iterable) -> List[str]:
        """
        Hash :class:`DataContainer`.current_ids, data inputs, and hyperparameters together.

        :param current_ids: current hashed ids (can be None if this function has not been called yet)
        :type current_ids: List[str]
        :param hyperparameters: step hyperparameters to hash with current ids
        :type hyperparameters: HyperparameterSamples
        :param data_inputs: data inputs to hash current ids for
        :type data_inputs: Iterable
        :return: the new hashed current ids
        :rtype: List[str]
        """
        raise NotImplementedError()


class HashlibMd5Hasher(BaseHasher):
    """
    Class to hash hyperparamters, and data input ids together using md5 algorithm from hashlib :
    `<https://docs.python.org/3/library/hashlib.html>`_

    The :class:`DataContainer` class uses the hashed values for its current ids.
    :class:`BaseStep` uses many :class:`BaseHasher` objects
    to hash hyperparameters, and data inputs ids together after each transform.

    .. seealso::
        :class`BaseHasher`,
        :class:`DataContainer`

    """

    def single_hash(self, current_id: str, hyperparameters: HyperparameterSamples) -> List[str]:
        """
        Hash summary id, and hyperparameters together.

        :param current_id: current hashed id
        :param hyperparameters: step hyperparameters to hash with current ids
        :type hyperparameters: HyperparameterSamples
        :return: the new hashed current id
        :rtype: str
        """
        m = hashlib.md5()

        current_hyperparameters_hash = hashlib.md5(
            str.encode(str(hyperparameters.to_flat_as_dict_primitive()))
        ).hexdigest()

        m.update(str.encode(str(current_id)))
        m.update(str.encode(current_hyperparameters_hash))

        return m.hexdigest()

    def hash(self, current_ids, hyperparameters, data_inputs: Any = None) -> List[str]:
        """
        Hash :class:`DataContainer`.current_ids, data inputs, and hyperparameters together
        using  `hashlib.md5 <https://docs.python.org/3/library/hashlib.html>`_

        :param current_ids: current hashed ids (can be None if this function has not been called yet)
        :type current_ids: List[str]
        :param hyperparameters: step hyperparameters to hash with current ids
        :type hyperparameters: HyperparameterSamples
        :param data_inputs: data inputs to hash current ids for
        :type data_inputs: Iterable
        :return: the new hashed current ids
        :rtype: List[str]
        """
        if current_ids is None:
            if isinstance(data_inputs, Iterable):
                current_ids = [str(i) for i in range(len(data_inputs))]
            else:
                current_ids = [str(0)]

        if len(hyperparameters) == 0:
            return current_ids

        hyperperams_dict = hyperparameters.to_flat_as_dict_primitive()
        current_hyperparameters_hash = hashlib.md5(str.encode(str(hyperperams_dict))).hexdigest()

        new_current_ids = []
        for current_id in current_ids:
            m = hashlib.md5()
            m.update(str.encode(current_id))
            m.update(str.encode(current_hyperparameters_hash))
            new_current_ids.append(m.hexdigest())

        return new_current_ids


class BaseSaver(ABC):
    """
    Any saver must inherit from this one. Some savers just save parts of objects, some save it all or what remains.
    Each :class`BaseStep` can potentially have multiple savers to make serialization possible.

    .. seealso::
        :func:`~neuraxle.base.BaseStep.save`,
        :func:`~neuraxle.base.BaseStep.load`
    """

    @abstractmethod
    def save_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Save step with execution context.

        :param step: step to save
        :type step: BaseStep
        :param context: execution context
        :type context: ExecutionContext
        :param save_savers:
        :type save_savers: bool
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def can_load(self, step: 'BaseStep', context: 'ExecutionContext'):
        """
        Returns true if we can load the given step with the given execution context.

        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def load_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Load step with execution context.

        :param step: step to load
        :param context: execution context to load from
        :return: loaded base step
        """
        raise NotImplementedError()

    def get_saved_files(self, step: 'BaseStep', context: 'ExecutionContext') -> List[Tuple[str, str]]:
        raise NotImplementedError()


class JoblibStepSaver(BaseSaver):
    """
    Saver that can save, or load a step with `joblib.load <https://joblib.readthedocs.io/en/latest/generated/joblib.load.html>`_,
    and `joblib.dump <https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html>`_.

    This saver is a good default saver when the object is
    already stripped out of things that would make it unserializable.

    It is the default stripped_saver for the :class:`ExecutionContext`.
    The stripped saver is the first to load the step, and the last to save the step.
    The saver receives a *stripped* version of the step so that it can be saved by joblib.

    .. seealso::
        :class:`BaseSaver`,
        :class:`ExecutionContext`
    """

    def get_saved_files(self, step: 'BaseStep', context: 'ExecutionContext') -> List[Tuple[str, str]]:
        return [(context.get_path(), self._create_step_path(step, context))]

    def can_load(self, step: 'BaseStep', context: 'ExecutionContext') -> bool:
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
            self._create_step_path(step, context)
        )

    def _create_step_path(self, step, context):
        """
        Create step path for the given context.

        :param context: execution context
        :type context: ExecutionContext
        :param step: step to save, or load
        :type step: BaseStep
        :return: path
        :rtype: str
        """
        return os.path.join(context.get_path(), '{0}.joblib'.format(step.name))

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

        path = self._create_step_path(step, context)
        dump(step, path)

        return step

    def load_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Load stripped step.

        :param step: stripped step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return:
        """
        loaded_step = load(self._create_step_path(step, context))

        # we need to keep the current steps in memory because they have been deleted before saving...
        # the steps that have not been saved yet need to be in memory while loading a truncable steps...
        if isinstance(loaded_step, TruncableSteps) and hasattr(step, 'steps'):
            loaded_step.steps = step.steps

        return loaded_step


class ExecutionMode(Enum):
    FIT_OR_FIT_TRANSFORM_OR_TRANSFORM = 'fit_or_fit_transform_or_transform'
    FIT_OR_FIT_TRANSFORM = 'fit_or_fit_transform'
    TRANSFORM = 'transform'
    FIT = 'fit'
    FIT_TRANSFORM = 'fit_transform'


class ExecutionContext:
    """
    Execution context object containing all of the pipeline hierarchy steps.
    First item in execution context parents is root, second is nested, and so on. This is like a stack.

    The execution context is used for fitted step saving, and caching :
        * :func:`~neuraxle.base.BaseStep.save`
        * :func:`~neuraxle.base.BaseStep.load`
        * :func:`~neuraxle.steps.caching.ValueCachingWrapper.handle_transform`
        * :func:`~neuraxle.steps.caching.ValueCachingWrapper.handle_fit_transform`

    .. seealso::
        :class:`BaseStep`,
        :class:`ValueCachingWrapper`
    """

    def __init__(
            self,
            root: str = DEFAULT_CACHE_FOLDER,
            execution_mode: ExecutionMode = None,
            stripped_saver: BaseSaver = None,
            parents=None
    ):
        if execution_mode is None:
            execution_mode = ExecutionMode.FIT_OR_FIT_TRANSFORM_OR_TRANSFORM
        self.execution_mode = execution_mode

        if stripped_saver is None:
            stripped_saver: BaseSaver = JoblibStepSaver()

        self.stripped_saver = stripped_saver
        self.root: str = root
        if parents is None:
            parents = []
        self.parents: List[BaseStep] = parents

    def get_execution_mode(self) -> ExecutionMode:
        return self.execution_mode

    def save(self, full_dump=False):
        """
        Save all unsaved steps in the parents of the execution context using :func:`~neuraxle.base.BaseStep.save`.
        This method is called from a step checkpointer inside a :class:`Checkpoint`.

        :param full_dump: save full pipeline dump to be able to load everything without source code (false by default).
        :type full_dump: bool
        :return:

        .. seealso::
            :class:`BaseStep`,
            :func:`~neuraxle.base.BaseStep.save`
        """
        while not self.empty():
            should_save_last_step = self.should_save_last_step()
            last_step = self.peek()
            if full_dump:
                should_save_last_step = True

            self.pop()
            if should_save_last_step:
                last_step.save(self, full_dump)

    def save_last(self) -> 'BaseStep':
        """
        Save only the last step in the execution context.

        .. seealso::
            :func:`~neuraxle.base.ExecutionContext.save`
        """
        last_step = self.peek()
        self.pop()
        return last_step.save(self, True)

    def should_save_last_step(self) -> bool:
        """
        Returns True if the last step should be saved.

        :return: if the last step should be saved
        :rtype: bool
        """
        if len(self.parents) > 0:
            return self.parents[-1].should_save()
        return False

    def pop_item(self) -> 'BaseStep':
        """
        Change the execution context to be the same as the latest parent context.

        :return:
        """
        return self.parents.pop()

    def pop(self) -> bool:
        """
        Pop the context. Returns True if it successfully popped an item from the parents list.

        :return: if an item has been popped
        :rtype: bool
        """
        if len(self) == 0:
            return False
        self.pop_item()
        return True

    def push(self, step: 'BaseStep') -> 'ExecutionContext':
        """
        Pushes a step in the parents of the execution context.

        :param step: step to add to the execution context
        :type step: BaseStep
        :return: self
        :rtype: ExecutionContext
        """
        return ExecutionContext(root=self.root, execution_mode=self.execution_mode, parents=self.parents + [step])

    def copy(self):
        return ExecutionContext(root=self.root, execution_mode=self.execution_mode, parents=copy(self.parents))

    def peek(self) -> 'BaseStep':
        """
        Get last parent.

        :return: the last parent base step
        :rtype: BaseStep
        """
        return self.parents[-1]

    def mkdir(self):
        """
        Creates the directory to save the last parent step.

        :return:
        """
        path = self.get_path()
        if not os.path.exists(path):
            os.makedirs(path)

    def get_path(self, with_root=True):
        """
        Creates the directory path for the current execution context.

        :return: current context path
        :rtype: str
        """
        parents_with_path = [self.root] if with_root else []
        parents_with_path += [p.name for p in self.parents]
        if len(parents_with_path) == 0:
            return '/'
        return os.path.join(*parents_with_path)

    def get_names(self):
        """
        Returns a list of the parent names.

        :return: list of parents step names
        :rtype: List[str]
        """
        return [p.name for p in self.parents]

    def empty(self):
        """
        Return True if the context has parent steps.

        :return: if parents len is 0
        :rtype: bool
        """
        return len(self) == 0

    def load(self, path: str) -> 'BaseStep':
        """
        Load full dump at the given path.

        :param path: pipeline step path
        :type path: str
        :return: loaded step
        :rtype: BaseStep

        .. seealso::
            :class:`FullDumpLoader`,
            :class:`Identity`
        """
        context_for_loading = self
        context_for_loading = context_for_loading.push(Identity(name=path))

        if os.sep in path:
            context_for_loading = context_for_loading.to_identity()
            path = path.split(os.sep)[-1]

        return FullDumpLoader(
            name=path,
            stripped_saver=self.stripped_saver
        ).load(context_for_loading, True)

    def to_identity(self) -> 'ExecutionContext':
        """
        Create a fake execution context containing only identity steps.
        Create the parents by using the path of the current execution context.

        :return: fake identity execution context
        :rtype: ExecutionContext

        .. seealso::
            :class:`FullDumpLoader`,
            :class:`Identity`
        """
        step_names = self.get_path(False).split(os.sep)

        parents = [
            Identity(name=name)
            for name in step_names
        ]

        return ExecutionContext(
            root=self.root,
            execution_mode=self.execution_mode,
            stripped_saver=self.stripped_saver,
            parents=parents
        )

    def get_all_saved_step_files(self) -> List[Tuple[str, str]]:
        saved_files = []
        last_step = self.peek()
        for saver in last_step.savers:
            saved_files.extend(saver.get_saved_files(last_step, self))

        saved_files.extend(self.stripped_saver.get_saved_files(last_step, self))

        return saved_files

    def __len__(self):
        return len(self.parents)


class BaseStep(ABC):
    """
    Base class for a pipeline step.

    Every step must implement :
        * :func:`~neuraxle.base.BaseStep.fit`
        * :func:`~neuraxle.base.BaseStep.fit_transform`
        * :func:`~neuraxle.base.BaseStep.transform`

    If a step is not fittable, you can inherit from :class:`NonFittableMixin`.
    If a step is not transformable, you can inherit from :class:`NonTransformableMixin`.
    A step should only change its state inside :func:`~neuraxle.base.BaseStep.fit` or :func:`~neuraxle.base.BaseStep.fit_transform`.

    Example usage :

    .. code-block:: python

        class MultiplyByN(NonFittableMixin, BaseStep):
            def __init__(self, multiply_by):
                NonFittableMixin.__init__(self)
                BaseStep.__init__(
                    self,
                    hyperparams=HyperparameterSamples({
                        'multiply_by': multiply_by
                    })
                )

            def transform(self, data_inputs):
                return data_inputs * self.hyperparams['multiply_by']

    Every step can be saved using its savers of type :class:`BaseSaver`. Some savers just save parts of objects, some save it all or what remains.
    Most step hash data inputs with hyperparams after every transformations to update the current ids inside the :class:`DataContainer`.

    Every step has handle methods that can be overridden to add side effects or change the execution flow based on the execution context, and the data container :
        * :func:`~neuraxle.base.BaseStep.handle_transform`
        * :func:`~neuraxle.base.BaseStep.handle_fit_transform`
        * :func:`~neuraxle.base.BaseStep.handle_fit`

    Every step has hyperparemeters, and hyperparameters spaces that can be set before the learning process begins.
    Hyperparameters can not only be passed in the constructor, but also be set by the pipeline that contains all of the steps :

    .. code-block:: python

        pipeline = Pipeline([
            SomeStep()
        ])

        pipeline.set_hyperparams(HyperparameterSamples({
            'learning_rate': 0.1,
            'SomeStep__learning_rate': 0.05
        }))

    .. note:: All heavy initialization logic should be done inside the *setup* method (e.g.: things inside GPU), and NOT in the constructor.
    .. seealso::
        :class:`neuraxle.pipeline.Pipeline`,
        :class:`NonFittableMixin`,
        :class:`NonTransformableMixin`,
        :class:`neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`BaseSaver`,
        :class:`BaseHasher`,
        :class:`DataContainer`
    """

    def __init__(
            self,
            hyperparams: HyperparameterSamples = None,
            hyperparams_space: HyperparameterSpace = None,
            name: str = None,
            savers: List[BaseSaver] = None,
            hashers: List[BaseHasher] = None
    ):
        if hyperparams is None:
            hyperparams = dict()
        if hyperparams_space is None:
            hyperparams_space = dict()
        if name is None:
            name = self.__class__.__name__
        if savers is None:
            savers = []
        if hashers is None:
            hashers = [HashlibMd5Hasher()]

        self.hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams)
        self.hyperparams = self.hyperparams.to_flat()

        self.hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space)
        self.hyperparams_space = self.hyperparams_space.to_flat()

        self.name: str = name

        self.savers: List[BaseSaver] = savers  # TODO: doc. First is the most stripped.
        self.hashers: List[BaseHasher] = hashers

        self.pending_mutate: ('BaseStep', str, str) = (None, None, None)
        self.is_initialized = False
        self.is_invalidated = True
        self.is_train: bool = True

    def summary_hash(self, data_container: DataContainer) -> str:
        """
        Hash data inputs, current ids, and hyperparameters together using self.hashers.
        This is used to create unique ids for the data checkpoints.

        :param data_container: data container
        :type data_container: DataContainer
        :return: hashed current ids
        :rtype: List[str]

        .. seealso::
            :class:`neuraxle.checkpoints.Checkpoint`
        """
        summary_id = data_container.summary_id
        for h in self.hashers:
            summary_id = h.single_hash(
                summary_id,
                self.hyperparams
            )

        return summary_id

    def hash(self, data_container: DataContainer) -> List[str]:
        """
        Hash data inputs, current ids, and hyperparameters together using self.hashers.
        This is used to create unique ids for the data checkpoints.

        :param data_container: data container
        :type data_container: DataContainer
        :return: hashed current ids
        :rtype: List[str]

        .. seealso::
            :class:`neuraxle.checkpoints.Checkpoint`
        """
        current_ids = data_container.current_ids
        for h in self.hashers:
            current_ids = h.hash(current_ids, self.hyperparams, data_container.data_inputs)

        return current_ids

    def setup(self) -> 'BaseStep':
        """
        Initialize the step before it runs. Only from here and not before that heavy things should be created
        (e.g.: things inside GPU), and NOT in the constructor.

        The setup method is called for each step before any fit, or fit_transform.

        :return: self
        :rtype: BaseStep
        """
        self.is_initialized = True
        return self

    def invalidate(self) -> 'BaseStep':
        """
        Invalidate step.

        :return: self
        :rtype: BaseStep
        """
        self.is_invalidated = True
        return self

    def teardown(self) -> 'BaseStep':
        """
        Teardown step after program execution. Inverse of setup, and it should clear memory.
        Override this method if you need to clear memory.

        :return: self
        :rtype: BaseStep
        """
        self.is_initialized = False
        return self

    def set_train(self, is_train: bool = True):
        """
        This method overrides the method of BaseStep to also consider the wrapped step as well as self.
        Set pipeline step mode to train or test.

        :param is_train: is training mode or not
        :type is_train: bool
        :return:

        .. seealso::
            :func:`BaseStep.set_train`
        """
        self.is_train = is_train
        return self

    def set_name(self, name: str):
        """
        Set the name of the pipeline step.

        :param name: a string.
        :type name: str
        :return: self

        .. note::
            A step name is the same value as the one in the keys of :any:`~neuraxle.pipeline.Pipeline.steps_as_tuple`
        """
        self.name = name
        self.is_invalidated = True
        return self

    def get_name(self) -> str:
        """
        Get the name of the pipeline step.

        :return: the name, a string.
        :rtype: str

        .. note:: A step name is the same value as the one in the keys of :class:`Pipeline`.steps_as_tuple
        """
        return self.name

    def get_savers(self) -> List[BaseSaver]:
        """
        Get the step savers of a pipeline step.

        :return: step savers
        :rtype: List[BaseSaver]

        .. seealso::
            :class:`BaseSaver`
        """
        return self.savers

    def set_savers(self, savers: List[BaseSaver]) -> 'BaseStep':
        """
        Set the step savers of a pipeline step.

        :return: self
        :rtype: BaseStep

        .. seealso::
            :class:`BaseSaver`
        """
        self.savers: List[BaseSaver] = savers
        return self

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> 'BaseStep':
        """
        Set the step hyperparameters.

        Example :

        .. code-block:: python

            step.set_hyperparams(HyperparameterSamples({
                'learning_rate': 0.10
            }))

        :param hyperparams: hyperparameters
        :return: self
        :rtype: BaseStep

        .. seealso::
            :class:`neuraxle.hyperparams.space.HyperparameterSamples`
        """
        self.is_invalidated = True
        self.hyperparams = HyperparameterSamples(hyperparams).to_flat()
        return self

    def update_hyperparams(self, hyperparams: HyperparameterSamples) -> 'BaseStep':
        """
        Update the step hyperparameters without removing the already-set hyperparameters.
        This can be useful to add more hyperparameters to the existing ones without flushing the ones that were already set.

        Example :

        .. code-block:: python

            step.set_hyperparams(HyperparameterSamples({
                'learning_rate': 0.10
                'weight_decay': 0.001
            }))

            step.update_hyperparams(HyperparameterSamples({
                'learning_rate': 0.01
            }))

            assert step.get_hyperparams()['learning_rate'] == 0.01
            assert step.get_hyperparams()['weight_decay'] == 0.001

        :param hyperparams: hyperparameters
        :type hyperparams: HyperparameterSamples
        :return: self
        :rtype: BaseStep

        .. seealso::
            :func:`~BaseStep.update_hyperparams`,
            :class:`neuraxle.hyperparams.space.HyperparameterSamples`
        """
        self.hyperparams.update(hyperparams)
        self.hyperparams = HyperparameterSamples(self.hyperparams).to_flat()
        return self

    def get_hyperparams(self) -> HyperparameterSamples:
        """
        Get step hyperparameters as :class:`neuraxle.hyperparams.space.HyperparameterSamples`.

        :return: step hyperparameters
        :rtype: HyperparameterSamples

        .. seealso::
            * :class:`neuraxle.hyperparams.space.HyperparameterSamples`
        """
        return self.hyperparams

    def set_params(self, **params) -> 'BaseStep':
        """
        Set step hyperparameters with a dictionary.

        Example :

        .. code-block:: python

            s.set_params(learning_rate=0.1)
            hyperparams = s.get_params()
            assert hyperparams == {"learning_rate": 0.1}

        :param **params: arbitrary number of arguments for hyperparameters
        :rtype: BaseStep

        .. seealso::
            :class:`neuraxle.hyperparams.space.HyperparameterSamples`
        """
        return self.set_hyperparams(HyperparameterSamples(params))

    def get_params(self) -> dict:
        """
        Get step hyperparameters as a flat primitive dict.

        Example :

        .. code-block:: python

            s.set_params(learning_rate=0.1)
            hyperparams = s.get_params()
            assert hyperparams == {"learning_rate": 0.1}

        :return: hyperparameters
        :rtype: dict

        .. seealso::
            :class:`neuraxle.hyperparams.space.HyperparameterSamples`
        """
        return self.get_hyperparams().to_flat_as_ordered_dict_primitive()

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        """
        Set step hyperparameters space.

        Example :

        .. code-block:: python

            step.set_hyperparams_space(HyperparameterSpace({
                'hp': RandInt(0, 10)
            }))

        :param hyperparams_space: hyperparameters space
        :type hyperparams_space: HyperparameterSpace
        :return: self
        :rtype: BaseStep

        .. seealso::
            :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
            :class:`neuraxle.hyperparams.distributions.HyperparameterDistribution`
        """
        self.is_invalidated = True
        self.hyperparams_space = HyperparameterSpace(hyperparams_space).to_flat()
        return self

    def get_hyperparams_space(self) -> HyperparameterSpace:
        """
        Get step hyperparameters space.

        Example :

        .. code-block:: python

            step.get_hyperparams_space()

        :return: step hyperparams space
        :rtype: HyperparameterSpace

        .. seealso::
            :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
            :class:`neuraxle.hyperparams.distributions.HyperparameterDistribution`
        """
        return self.hyperparams_space

    def handle_inverse_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Override this to add side effects or change the execution flow before (or after) calling :func:`~neuraxle.base.BaseStep.inverse_transform`.
        The default behavior is to rehash current ids with the step hyperparameters.

        :param data_container: the data container to inverse transform
        :param context: execution context
        :return: data_container

        .. seealso::
            :class:`DataContainer`,
            :class:`neuraxle.pipeline.Pipeline`
        """
        self.is_invalidated = True

        processed_outputs = self.inverse_transform(data_container.data_inputs)
        data_container.set_data_inputs(processed_outputs)

        current_ids = self.hash(data_container)
        data_container.set_current_ids(current_ids)

        return data_container

    def apply_method(self, method: Callable, *kargs, **kwargs) -> 'BaseStep':
        """
        Apply a method to a step and its children.

        :param method: method to call with self
        :param kargs: any additional arguments to be passed to the method
        :param kwargs: any additional positional arguments to be passed to the method
        :return: self (not a new step)
        :rtype: BaseStep
        """
        method(self, *kargs, **kwargs)
        return self

    def apply(self, method_name: str, *kargs, **kwargs) -> 'BaseStep':
        """
        Apply a method to a step and its children.

        :param method_name: method name that need to be called on all steps
        :param kargs: any additional arguments to be passed to the method
        :param kwargs: any additional positional arguments to be passed to the method
        :return: self (not a new step)
        :rtype: BaseStep
        """
        if hasattr(self, method_name) and callable(getattr(self, method_name)):
            getattr(self, method_name)(*kargs, **kwargs)

        return self

    def get_step_by_name(self, name):
        if self.name == name:
            return self
        return None

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseStep':
        """
        Override this to add side effects or change the execution flow before (or after) calling :func:`~neuraxle.base.BaseStep.fit`.
        The default behavior is to rehash current ids with the step hyperparameters.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)

        .. seealso::
            :class:`DataContainer`,
            :class:`neuraxle.pipeline.Pipeline`
        """
        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_fit(data_container, context)

        new_self = self._fit_data_container(data_container, context)

        return new_self

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> (
            'BaseStep', DataContainer):
        """
        Override this to add side effects or change the execution flow before (or after) calling * :func:`~neuraxle.base.BaseStep.fit_transform`.
        The default behavior is to rehash current ids with the step hyperparameters.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)
        """
        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_fit_transform(data_container, context)

        new_self, data_container = self._fit_transform_data_container(data_container, context)

        data_container = self._did_fit_transform(data_container, context)
        data_container = self._did_process(data_container, context)

        return new_self, data_container

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Override this to add side effects or change the execution flow before (or after) calling * :func:`~neuraxle.base.BaseStep.transform`.
        The default behavior is to rehash current ids with the step hyperparameters.

        :param data_container: the data container to transform
        :param context: execution context
        :return: transformed data container
        """
        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_transform_data_container(data_container, context)

        data_container = self._transform_data_container(data_container, context)

        data_container = self._did_transform(data_container, context)
        data_container = self._did_process(data_container, context)

        return data_container

    def _will_fit(self, data_container: DataContainer, context: ExecutionContext) -> (DataContainer, ExecutionContext):
        """
        Before fit is called, apply side effects on the step, the data container, or the execution context.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        :rtype: (DataContainer, ExecutionContext)
        """
        self.is_invalidated = True
        return data_container, context.push(self)

    def _did_fit(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Apply side effects before fit is called.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        :rtype: (DataContainer, ExecutionContext)
        """
        return data_container

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseStep':
        """
        Fit data container.

        :param data_container: data container
        :param context: execution context
        :return: (fitted self, data container)
        :rtype: (BaseStep, DataContainer)
        """
        return self.fit(data_container.data_inputs, data_container.expected_outputs)

    def _will_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> (
            DataContainer, ExecutionContext):
        """
        Apply side effects before fit_transform is called.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        :rtype: (DataContainer, ExecutionContext)
        """
        self.is_invalidated = True
        return data_container, context.push(self)

    def _did_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Apply side effects after fit transform.

        :param data_container: data container
        :param context: execution context
        :return: (fitted self, data container)
        :rtype: (BaseStep, DataContainer)
        """
        return data_container

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (
            'BaseStep', DataContainer):
        """
        Fit transform data container.

        :param data_container: data container
        :param context: execution context
        :return: (fitted self, data container)
        :rtype: (BaseStep, DataContainer)
        """
        new_self, out = self.fit_transform(data_container.data_inputs, data_container.expected_outputs)
        data_container.set_data_inputs(out)

        return new_self, data_container

    def _will_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (
            'BaseStep', DataContainer):
        """
        Apply side effects before transform.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        :rtype: (DataContainer, ExecutionContext)
        """
        return data_container, context.push(self)

    def _will_process(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Apply side effects before any step method.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        :rtype: (DataContainer, ExecutionContext)
        """
        self.setup()
        return data_container, context

    def _did_process(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Apply side effects after any step method.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        :rtype: (DataContainer, ExecutionContext)
        """
        data_container = self.hash_data_container(data_container)
        return data_container

    def _did_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Apply side effects after transform.

        :param data_container: data container
        :param context: execution context
        :return: data container
        :rtype: DataContainer
        """
        return data_container

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Transform data container.

        :param data_container: data container
        :param context: execution context
        :return: data container
        :rtype: DataContainer
        """
        out = self.transform(data_container.data_inputs)
        data_container.set_data_inputs(out)

        return data_container

    def hash_data_container(self, data_container):
        """
        Hash data container using self.hashers.

        #. Hash current ids with hyperparams.
        #. Hash summary id with hyperparams.

        :param data_container: the data container to transform
        :type data_container: DataContainer
        :return: transformed data container
        :rtype: DataContainer
        """
        current_ids = self.hash(data_container)
        data_container.set_current_ids(current_ids)

        if data_container.summary_id is None:
            data_container.set_summary_id(
                data_container.hash_summary()
            )

        summary_id = self.summary_hash(data_container)
        data_container.set_summary_id(summary_id)

        return data_container

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        """
        Fit, and transform step with the given data inputs, and expected outputs.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on
        :return: (fitted self, tranformed data inputs)
        :rtype: Tuple[BaseStep, Any]
        """
        self.is_invalidated = True

        new_self = self.fit(data_inputs, expected_outputs)
        out = new_self.transform(data_inputs)

        return new_self, out

    @abstractmethod
    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        """
        Fit step with the given data inputs, and expected outputs.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on
        :return: fitted self
        :rtype: BaseStep
        """
        raise NotImplementedError(
            "TODO: Implement this method in {}, or have this class inherit from the NonFittableMixin.".format(
                self.__class__.__name__))

    @abstractmethod
    def transform(self, data_inputs):
        """
        Transform given data inputs.

        :param data_inputs: data inputs
        :return: transformed data inputs
        :rtype: Any
        """
        raise NotImplementedError(
            "TODO: Implement this method in {}, or have this class inherit from the NonTransformableMixin.".format(
                self.__class__.__name__))

    def inverse_transform(self, processed_outputs):
        """
        Inverse Transform the given transformed data inputs.

        :func:`~neuraxle.base.BaseStep.mutate` or :func:`~neuraxle.base.BaseStep.reverse` can be called to change the default transform behavior :

        .. code-block:: python

            p = Pipeline([MultiplyBy()])

            _in = np.array([1, 2])

            _out = p.transform(_in)

            _regenerated_in = reversed(p).transform(_out)

            assert np.array_equal(_regenerated_in, _in)

        :param processed_outputs: processed data inputs
        :return: inverse transformed processed outputs
        :rtype: Any
        """
        raise NotImplementedError("TODO: Implement this method in {}.".format(self.__class__.__name__))

    def predict(self, data_input):
        """
        Predict data input expected output using transform method.
        This is simply a shorthand method that does the same thing as func:`~.transform`.

        :param data_input: data input to predict
        :return: prediction
        :rtype: Any
        """
        return self.transform(data_input)

    def should_save(self) -> bool:
        """
        Returns true if the step should be saved.
        If the step has been initialized and invalidated, then it must be saved.

        A step is invalidated when any of the following things happen :
            * a mutation has been performed on the step : func:`~.mutate`
            * an hyperparameter has changed func:`~.set_hyperparams`
            * an hyperparameter space has changed func:`~.set_hyperparams_space`
            * a call to the fit method func:`~.handle_fit`
            * a call to the fit_transform method func:`~.handle_fit_transform`
            * the step name has changed func:`~neuraxle.base.BaseStep.set_name`

        :return: if the step should be saved
        :rtype: bool
        """
        return self.is_invalidated and self.is_initialized

    def save(self, context: ExecutionContext, full_dump=False) -> 'BaseStep':
        """
        Save step using the execution context to create the directory to save the step into.
        The saving happens by looping through all of the step savers in the reversed order.

        Some savers just save parts of objects, some save it all or what remains.
        The :class:`ExecutionContext`.stripped_saver has to be called last because it needs a
        stripped version of the step.

        :param context: context to save from
        :type context: ExecutionContext
        :param full_dump: save full pipeline dump to be able to load everything without source code (false by default).
        :type full_dump: bool
        :return: self
        :rtype: BaseStep

        .. seealso::
            :class:`ExecutionContext`,
            :class:`BaseSaver`
        """
        context = context.push(self)
        self.is_invalidated = False

        if full_dump:
            # initialize and invalidate steps to make sure that all steps will be saved
            self.apply_method(lambda step: step.setup() if not step.is_initialized else None)
            self.apply_method(lambda step: step.invalidate())

        return self._save_step(context)

    def _save_step(self, context):
        context.mkdir()
        stripped_step = copy(self)

        # A final "visitor" saver will save anything that
        # wasn't saved customly after stripping the rest.
        savers_with_provided_default_stripped_saver = [context.stripped_saver] + self.savers

        for saver in reversed(savers_with_provided_default_stripped_saver):
            # Each saver strips the step a bit more if needs be.
            stripped_step = saver.save_step(stripped_step, context)

        return stripped_step

    def load(self, context: ExecutionContext, full_dump=False) -> 'BaseStep':
        """
        Load step using the execution context to create the directory of the saved step.
        Warning:

        :param context: execution context to load step from
        :param full_dump: save full dump bool
        :type full_dump: bool
        :return: loaded step

        .. warning::
            Please do not override this method because on loading it is an identity
            step that will load whatever step you coded.

        .. seealso::
            :class:`ExecutionContext`,
            :class:`BaseSaver`
        """
        context = context.push(self)

        savers = [context.stripped_saver] + self.savers
        return self._load_step(context, savers)

    def _load_step(self, context, savers):
        # A final "visitor" saver might reload anything that wasn't saved customly after stripping the rest.
        loaded_self = self
        for saver in savers:
            # Each saver unstrips the step a bit more if needed
            if saver.can_load(loaded_self, context):
                loaded_self = saver.load_step(loaded_self, context)
            else:
                warnings.warn(
                    'Cannot Load Step {0} ({1}:{2}) With Step Saver {3}.'.format(context.get_path(), self.name,
                                                                                 self.__class__.__name__,
                                                                                 saver.__class__.__name__))
                break
        return loaded_self

    def meta_fit(self, X_train, y_train, metastep: 'MetaStepMixin'):
        """
        Uses a meta optimization technique (AutoML) to find the best hyperparameters in the given
        hyperparameter space.

        Usage: ``p = p.meta_fit(X_train, y_train, metastep=RandomSearch(n_iter=10, scoring_function=r2_score, higher_score_is_better=True))``

        Call ``.mutate(new_method="inverse_transform", method_to_assign_to="transform")``, and the
        current estimator will become

        :param X_train: data_inputs.
        :param y_train: expected_outputs.
        :param metastep: a metastep, that is, a step that can sift through the hyperparameter space of another estimator.
        :return: your best self.
        """
        metastep.set_step(self)
        metastep = metastep.fit(X_train, y_train)
        best_step = metastep.get_best_model()
        return best_step

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=True) -> 'BaseStep':
        """
        Replace the "method_to_assign_to" method by the "new_method" method, IF the present object has no pending calls to
        ``.will_mutate_to()`` waiting to be applied. If there is a pending call, the pending call will override the
        methods specified in the present call. If the change fails (such as if the new_method doesn't exist), then
        a warning is printed (optional). By default, there is no pending ``will_mutate_to`` call.

        This could for example be useful within a pipeline to apply ``inverse_transform`` to every pipeline steps, or
        to assign ``predict_probas`` to ``predict``, or to assign "inverse_transform" to "transform" to a reversed pipeline.

        :param new_method: the method to replace transform with, if there is no pending ``will_mutate_to`` call.
        :param method_to_assign_to: the method to which the new method will be assigned to, if there is no pending ``will_mutate_to`` call.
        :param warn: (verbose) wheter or not to warn about the inexistence of the method.
        :return: self, a copy of self, or even perhaps a new or different BaseStep object.
        """
        self.is_invalidated = True
        pending_new_base_step, pending_new_method, pending_method_to_assign_to = self.pending_mutate

        # Use everything that is pending if they are not none (ternaries).
        new_base_step = pending_new_base_step if pending_new_base_step is not None else copy(self)
        new_method = pending_new_method if pending_new_method is not None else new_method
        method_to_assign_to = pending_method_to_assign_to if pending_method_to_assign_to is not None else method_to_assign_to

        # We set "new_method" in place of "method_to_affect" to a copy of self:
        try:
            # 1. get new method's reference
            new_method = getattr(new_base_step, new_method)

            # 2. delete old method
            try:
                delattr(new_base_step, method_to_assign_to)
            except AttributeError as e:
                pass

            # 3. assign new method to old method
            setattr(new_base_step, method_to_assign_to, new_method)
            self.is_invalidated = True

        except AttributeError as e:
            if warn:
                import warnings
                warnings.warn(e)

        return new_base_step

    def will_mutate_to(
            self, new_base_step: 'BaseStep' = None, new_method: str = None, method_to_assign_to: str = None
    ) -> 'BaseStep':
        """
        This will change the behavior of ``self.mutate(<...>)`` such that when mutating, it will return the
        presently provided new_base_step BaseStep (can be left to None for self), and the ``.mutate`` method
        will also apply the ``new_method`` and the  ``method_to_affect``, if they are not None, and after changing
        the object to new_base_step.

        This can be useful if your pipeline requires unsupervised pretraining. For example:

        .. code-block:: python

            X_pretrain = ...
            X_train = ...

            p = Pipeline(
                SomePreprocessing(),
                SomePretrainingStep().will_mutate_to(new_base_step=SomeStepThatWillUseThePretrainingStep),
                Identity().will_mutate_to(new_base_step=ClassifierThatWillBeUsedOnlyAfterThePretraining)
            )
            # Pre-train the pipeline
            p = p.fit(X_pretrain, y=None)

            # This will leave `SomePreprocessing()` untouched and will affect the two other steps.
            p = p.mutate(new_method="transform", method_to_affect="transform")

            # Pre-train the pipeline
            p = p.fit(X_train, y_train)  # Then fit the classifier and other new things

        :param new_base_step: if it is not None, upon calling ``mutate``, the object it will mutate to will be this provided new_base_step.
        :type new_base_step: BaseStep
        :param method_to_assign_to: if it is not None, upon calling ``mutate``, the method_to_affect will be the one that is used on the provided new_base_step.
        :type method_to_assign_to: str
        :param new_method: if it is not None, upon calling ``mutate``, the new_method will be the one that is used on the provided new_base_step.
        :type new_method: str
        :return: self
        :rtype: BaseStep
        """
        self.is_invalidated = True

        if new_method is None or method_to_assign_to is None:
            new_method = method_to_assign_to = "transform"  # No changes will be applied (transform will stay transform).

        self.pending_mutate = (new_base_step, new_method, method_to_assign_to)

        return self

    def tosklearn(self):
        class NeuraxleToSKLearnPipelineWrapper(BaseEstimator):
            def __init__(self, neuraxle_step):
                self.p: Union[BaseStep, TruncableSteps] = neuraxle_step

            def set_params(self, **params) -> BaseEstimator:
                self.p.set_hyperparams(HyperparameterSpace(params))
                return self

            def get_params(self, deep=True):
                neuraxle_params = HyperparameterSamples(self.p.get_hyperparams()).to_flat_as_dict_primitive()
                return neuraxle_params

            def get_params_space(self, deep=True):
                neuraxle_params = HyperparameterSpace(self.p.get_hyperparams_space()).to_flat_as_dict_primitive()
                return neuraxle_params

            def fit(self, **args) -> BaseEstimator:
                self.p = self.p.fit(**args)
                return self

            def transform(self, **args):
                return self.p.transform(**args)

            def fit_transform(self, **args) -> Any:
                self.p, out = self.p.fit_transform(**args)
                # Careful: 1 return value.
                return out

            def inverse_transform(self, **args):
                return self.p.reverse().transform(**args)

            def predict(self, **args):
                return self.p.transform(**args)

            def __repr__(self):
                return self.__class__.__name__ + "(" + self.p.__repr__() + ")"

            def __str__(self):
                return self.__repr__()

        return NeuraxleToSKLearnPipelineWrapper(self)

    def reverse(self) -> 'BaseStep':
        """
        The object will mutate itself such that the ``.transform`` method (and of all its underlying objects
        if applicable) be replaced by the ``.inverse_transform`` method.

        Note: the reverse may fail if there is a pending mutate that was set earlier with ``.will_mutate_to``.

        :return: a copy of self, reversed. Each contained object will also have been reversed if self is a pipeline.
        .. seealso::
            :func:`~neuraxle.base.BaseStep.__reversed__`,
            :func:`~neuraxle.base.BaseStep.inverse_transform`
        """
        return self.mutate(new_method="inverse_transform", method_to_assign_to="transform")

    def __reversed__(self) -> 'BaseStep':
        """
        The object will mutate itself such that the ``.transform`` method (and of all its underlying objects
        if applicable) be replaced by the ``.inverse_transform`` method.

        Note: the reverse may fail if there is a pending mutate that was set earlier with ``.will_mutate_to``.

        :return: a copy of self, reversed. Each contained object will also have been reversed if self is a pipeline.
        """
        return self.reverse()

    def __repr__(self):

        output = self.__class__.__name__ + "(\n\tname=" + self.name + "," + "\n\thyperparameters=" + pprint.pformat(
            self.hyperparams) + "\n)"

        return output

    def __str__(self):
        return self.__repr__()


class MetaStepMixin:
    """
    A class to represent a step that wraps another step. It can be used for many things.

    For example, :class:`ForEachDataInputs` adds a loop before any calls to the wrapped step :

    .. code-block:: python

        class ForEachDataInputs(MetaStepMixin, BaseStep):
            def __init__(
                self,
                wrapped: BaseStep
            ):
                BaseStep.__init__(self)
                MetaStepMixin.__init__(self, wrapped)

            def fit(self, data_inputs, expected_outputs=None):
                if expected_outputs is None:
                    expected_outputs = [None] * len(data_inputs)

                for di, eo in zip(data_inputs, expected_outputs):
                    self.wrapped = self.wrapped.fit(di, eo)

                return self

            def transform(self, data_inputs):
                outputs = []
                for di in data_inputs:
                    output = self.wrapped.transform(di)
                    outputs.append(output)

            return outputs

            def fit_transform(self, data_inputs, expected_outputs=None):
                if expected_outputs is None:
                    expected_outputs = [None] * len(data_inputs)

                outputs = []
                for di, eo in zip(data_inputs, expected_outputs):
                    self.wrapped, output = self.wrapped.fit_transform(di, eo)
                outputs.append(output)

                return self, outputs

    .. seealso::
        :class:`ForEachDataInputs`,
        :class:`MetaSKLearnWrapper`,
        :class:`RandomSearch`,
        :class:`BaseCrossValidation`,
        :class:`ValueCachingWrapper`,
        :class:`StepClonerForEachDataInput`
    """

    def __init__(
            self,
            wrapped: BaseStep = None
    ):
        self.wrapped: BaseStep = wrapped

        if not hasattr(self, 'savers'):
            warnings.warn('Please initialize Mixins in the reverse order. MetaStepMixin should be initialized after BaseStep for {}. Appending the MetaStepJoblibStepSaver to the savers.'.format(self.wrapped.name))
            self.savers = [MetaStepJoblibStepSaver()]
        else:
            self.savers.append(MetaStepJoblibStepSaver())

    def setup(self) -> BaseStep:
        """
        Initialize step before it runs. Also initialize the wrapped step.

        :return: self
        :rtype: BaseStep
        """
        BaseStep.setup(self)
        self.wrapped.setup()
        self.is_initialized = True
        return self

    def set_train(self, is_train: bool = True):
        """
        Set pipeline step mode to train or test. Also set wrapped step mode to train or test.

        For instance, you can add a simple if statement to direct to the right implementation:

        .. code-block:: python

            def transform(self, data_inputs):
                if self.is_train:
                    self.transform_train_(data_inputs)
                else:
                    self.transform_test_(data_inputs)

            def fit_transform(self, data_inputs, expected_outputs):
                if self.is_train:
                    self.fit_transform_train_(data_inputs, expected_outputs)
                else:
                    self.fit_transform_test_(data_inputs, expected_outputs)

        :param is_train: bool
        :return:
        """
        self.is_train = is_train
        self.wrapped.set_train(is_train)
        return self

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        """
        Set step hyperparameters, and wrapped step hyperparams with the given hyperparams.

        Example :

        .. code-block:: python

            step.set_hyperparams(HyperparameterSamples({
                'learning_rate': 0.10
                'wrapped__learning_rate': 0.10 # this will set the wrapped step 'learning_rate' hyperparam
            }))

        :param hyperparams: hyperparameters
        :type hyperparams: HyperparameterSamples
        :return: self
        :rtype: BaseStep

        .. seealso::
            :class:`HyperparameterSamples`
        """
        self.is_invalidated = True

        hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams.items():
            if name == self.wrapped.name:
                self.wrapped.set_hyperparams(hparams)
            else:
                remainders[name] = hparams

        self.hyperparams = HyperparameterSamples(remainders)

        return self

    def update_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        """
        Update the step, and the wrapped step hyperparams without removing the already set hyperparameters.
        Please refer to :func:`~BaseStep.update_hyperparams`.

        :param hyperparams: hyperparameters
        :type hyperparams: HyperparameterSamples
        :return: self
        :rtype: BaseStep

        .. seealso::
            :func:`~BaseStep.update_hyperparams`,
            :class:`HyperparameterSamples`
        """
        self.is_invalidated = True

        hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams.items():
            if name == self.wrapped.name:
                self.wrapped.update_hyperparams(hparams)
            else:
                remainders[name] = hparams

        self.hyperparams.update(remainders)

        return self

    def get_hyperparams(self) -> HyperparameterSamples:
        """
        Get step hyperparameters as :class:`HyperparameterSamples` with flattened hyperparams.

        :return: step hyperparameters
        :rtype: HyperparameterSamples

        .. seealso::
            :class:`HyperparameterSamples`
        """
        return HyperparameterSamples({
            **self.hyperparams.to_flat_as_dict_primitive(),
            self.wrapped.name: self.wrapped.get_hyperparams().to_flat_as_dict_primitive()
        }).to_flat()

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        """
        Set meta step and wrapped step hyperparams space using the given hyperparams space.

        :param hyperparams_space: ordered dict containing all hyperparameter spaces
        :type hyperparams_space: HyperparameterSpace

        :return: self
        """
        self.is_invalidated = True

        hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams_space.items():
            if name == self.wrapped.name:
                self.wrapped.set_hyperparams_space(hparams)
            else:
                remainders[name] = hparams

        self.hyperparams_space = HyperparameterSpace(remainders)

        return self

    def get_hyperparams_space(self) -> HyperparameterSpace:
        """
        Get meta step and wrapped step hyperparams as a flat hyperparameter space

        :return: hyperparameters_space
        :rtype: HyperparameterSpace
        """
        return HyperparameterSpace({
            **self.hyperparams_space.to_flat_as_dict_primitive(),
            self.wrapped.name: self.wrapped.get_hyperparams_space().to_flat_as_dict_primitive()
        }).to_flat()

    def set_step(self, step: BaseStep) -> BaseStep:
        """
        Set wrapped step to the given step.

        :param step: new wrapped step
        :type step: BaseStep
        :return: self
        :rtype: BaseStep
        """
        self.is_invalidated = True
        self.wrapped: BaseStep = step
        return self

    def get_step(self) -> BaseStep:
        """
        Get wrapped step

        :return: self.wrapped
        :rtype: BaseStep
        """
        return self.wrapped

    def get_best_model(self) -> BaseStep:
        return self.best_model

    def _fit_transform_data_container(self, data_container, context):
        self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
        return self, data_container

    def _fit_data_container(self, data_container, context):
        self.wrapped = self.wrapped.handle_fit(data_container, context)
        return self

    def _transform_data_container(self, data_container, context):
        data_container = self.wrapped.handle_transform(data_container, context)
        return data_container

    def fit_transform(self, data_inputs, expected_outputs):
        self.wrapped, data_inputs = self.wrapped.fit_transform(data_inputs, expected_outputs)
        return self, data_inputs

    def fit(self, data_inputs, expected_outputs):
        self.wrapped = self.wrapped.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        data_inputs = self.wrapped.transform(data_inputs)
        return data_inputs

    def should_resume(self, data_container: DataContainer, context: ExecutionContext):
        context = context.push(self)
        if isinstance(self.wrapped, ResumableStepMixin) and self.wrapped.should_resume(data_container, context):
            return True
        return False

    def apply(self, method_name: str, *kargs, **kwargs) -> 'BaseStep':
        """
        Apply the method name to the meta step and its wrapped step.

        :param method_name: method name that need to be called on all steps
        :param kargs: any additional arguments to be passed to the method
        :param kwargs: any additional positional arguments to be passed to the method
        :return: self (not a new step)
        :rtype: BaseStep
        """
        BaseStep.apply(self, method_name, *kargs, **kwargs)
        self.wrapped.apply(method_name, *kargs, **kwargs)
        return self

    def apply_method(self, method: Callable, *kargs, **kwargs) -> 'BaseStep':
        """
        Apply method to the meta step and its wrapped step.

        :param method: method to call with self
        :param kargs: any additional arguments to be passed to the method
        :param kwargs: any additional positional arguments to be passed to the method
        :return: self (not a new step)
        :rtype: BaseStep
        """
        BaseStep.apply_method(self, method, *kargs, **kwargs)
        self.wrapped = self.wrapped.apply_method(method, *kargs, **kwargs)
        return self

    def get_step_by_name(self, name):
        if self.wrapped.name == name:
            return self.wrapped
        return self.wrapped.get_step_by_name(name)

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=True) -> 'BaseStep':
        """
        Mutate self, and self.wrapped. Please refer to :func:`~neuraxle.base.BaseStep.mutate` for more information.

        :param new_method: the method to replace transform with, if there is no pending ``will_mutate_to`` call.
        :param method_to_assign_to: the method to which the new method will be assigned to, if there is no pending ``will_mutate_to`` call.
        :param warn: (verbose) wheter or not to warn about the inexistence of the method.
        :return: self, a copy of self, or even perhaps a new or different BaseStep object.
        """
        new_self = BaseStep.mutate(self, new_method, method_to_assign_to, warn)
        self.wrapped = self.wrapped.mutate(new_method, method_to_assign_to, warn)

        return new_self

    def will_mutate_to(
            self, new_base_step: 'BaseStep' = None, new_method: str = None, method_to_assign_to: str = None
    ) -> 'BaseStep':
        """
        Add pending mutate self, self.wrapped. Please refer to :func:`~neuraxle.base.BaseStep.will_mutate_to` for more information.

        :param new_base_step: if it is not None, upon calling ``mutate``, the object it will mutate to will be this provided new_base_step.
        :type new_base_step: BaseStep
        :param method_to_assign_to: if it is not None, upon calling ``mutate``, the method_to_affect will be the one that is used on the provided new_base_step.
        :type method_to_assign_to: str
        :param new_method: if it is not None, upon calling ``mutate``, the new_method will be the one that is used on the provided new_base_step.
        :type new_method: str
        :return: self
        :rtype: BaseStep
        """
        new_self = BaseStep.will_mutate_to(self, new_base_step, new_method, method_to_assign_to)
        return new_self

    def __repr__(self):
        output = self.__class__.__name__ + "(\n\twrapped=" + repr(
            self.wrapped) + "," + "\n\thyperparameters=" + pprint.pformat(
            self.hyperparams) + "\n)"
        return output


class MetaStepJoblibStepSaver(JoblibStepSaver):
    """
    Custom saver for meta step mixin.
    """
    def __init__(self):
        JoblibStepSaver.__init__(self)

    def save_step(self, step: 'MetaStepMixin', context: ExecutionContext) -> MetaStepMixin:
        """
        Save MetaStepMixin.

        #. Save wrapped step.
        #. Strip wrapped step form the meta step mixin.
        #. Save meta step with wrapped step savers.

        :param step: meta step to save
        :type step: MetaStepMixin
        :param context: execution context
        :type context: ExecutionContext
        :return:
        """
        # First, save the wrapped step savers
        wrapped_step_savers = []
        if step.wrapped.should_save():
            wrapped_step_savers.extend(step.wrapped.get_savers())
        else:
            wrapped_step_savers.append(None)

        # Second, save the wrapped step
        step.wrapped.save(context)

        step.wrapped_step_name_and_savers = (step.wrapped.name, wrapped_step_savers)

        # Third, strip the wrapped step from the meta step
        del step.wrapped

        return step

    def load_step(self, step: 'MetaStepMixin', context: ExecutionContext) -> 'MetaStepMixin':
        """
        Load MetaStepMixin.

        #. Loop through all of the sub steps savers, and only load the sub steps that have been saved.
        #. Refresh steps

        :param step: step to load
        :type step: BaseStep
        :param context: execution context
        :type context: ExecutionContext
        :return: loaded truncable steps
        :rtype: TruncableSteps
        """
        step_name, savers = step.wrapped_step_name_and_savers

        if savers is None:
            # keep wrapped step as it is if it hasn't been saved
            pass
        else:
            # load each sub step with their savers
            sub_step_to_load = Identity(name=step_name, savers=savers)
            sub_step = sub_step_to_load.load(context)
            step.wrapped = sub_step

        return step


NamedTupleList = List[Union[Tuple[str, 'BaseStep'], 'BaseStep']]


class ForceAlwaysHandleMixin:
    """
    A pipeline step that requires the implementation only of handler methods :

        - handle_transform
        - handle_fit_transform
        - handle_fit

    .. seealso::
        :class:`BaseStep`
    """

    @abstractmethod
    def handle_fit(self, data_container: DataContainer, context: ExecutionContext):
        raise NotImplementedError('Must implement handle_fit in {0}'.format(self.name))

    @abstractmethod
    def handle_transform(self, data_container: DataContainer, context: ExecutionContext):
        raise NotImplementedError('Must implement handle_transform in {0}'.format(self.name))

    @abstractmethod
    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        raise NotImplementedError('Must implement handle_fit_transform in {0}'.format(self.name))

    def transform(self, data_inputs) -> 'ForceAlwaysHandleMixin':
        raise Exception('Transform method is not supported for {0}, because it inherits from ForceHandleMixin. Please use handle_transform instead.'.format(self.name))

    def fit(self, data_inputs, expected_outputs=None) -> 'ForceAlwaysHandleMixin':
        raise Exception('Fit method is not supported for {0}, because it inherits from ForceHandleMixin. Please use handle_fit instead.'.format(self.name))

    def fit_transform(self, data_inputs, expected_outputs=None) -> 'ForceAlwaysHandleMixin':
        raise Exception('Fit transform method is not supported for {0}, because it inherits from ForceHandleMixin. Please use handle_fit_transform instead.'.format(self.name))


class NonFittableMixin:
    """
    A pipeline step that requires no fitting: fitting just returns self when called to do no action.
    Note: fit methods are not implemented
    """

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_transform_data_container(data_container, context)

        data_container = self._transform_data_container(data_container, context)

        data_container = self._did_transform(data_container, context)
        data_container = self._did_process(data_container, context)

        return self, data_container

    def fit(self, data_inputs, expected_outputs=None) -> 'NonFittableMixin':
        """
        Don't fit.

        :param data_inputs: the data that would normally be fitted on.
        :param expected_outputs: the data that would normally be fitted on.
        :return: self
        """
        return self


class NonTransformableMixin:
    """
    A pipeline step that has no effect at all but to return the same data without changes.
    Transform method is automatically implemented as changing nothing.

    Example :

    .. code-block:: python

        class PrintOnFit(NonTransformableMixin, BaseStep):
            def __init__(self):
                BaseStep.__init__(self)

            def fit(self, data_inputs, expected_outputs=None) -> 'FitCallbackStep':
                print((data_inputs, expected_outputs))
                return self

    .. note::
        fit methods are not implemented
    """

    def transform(self, data_inputs):
        """
        Do nothing - return the same data.

        :param data_inputs: the data to process
        :return: the ``data_inputs``, unchanged.
        """
        return data_inputs

    def inverse_transform(self, processed_outputs):
        """
        Do nothing - return the same data.

        :param processed_outputs: the data to process
        :return: the ``processed_outputs``, unchanged.
        """
        return processed_outputs


class TruncableJoblibStepSaver(JoblibStepSaver):
    """
    Step saver for a TruncableSteps.
    TruncableJoblibStepSaver saves, and loads all of the sub steps using their savers.

    .. seealso::
        :class:`JoblibStepSaver`,
        :class:`TruncableSteps`,
        :class:`BaseSaver`
    """

    def __init__(self):
        JoblibStepSaver.__init__(self)

    def save_step(self, step: 'TruncableSteps', context: ExecutionContext):
        """
        #. Loop through all the steps, and save the ones that need to be saved.
        #. Add a new property called sub step savers inside truncable steps to be able to load sub steps when loading.
        #. Strip steps from truncable steps at the end.

        :param step: step to save
        :type step: TruncableSteps
        :param context: execution context
        :type context: ExecutionContext
        :return:
        """

        # First, save all of the sub steps with the right execution context.
        sub_steps_savers = []
        for i, (_, sub_step) in enumerate(step):
            if sub_step.should_save():
                sub_steps_savers.append((step[i].name, step[i].get_savers()))
                sub_step.save(context)
            else:
                sub_steps_savers.append((step[i].name, None))

        step.sub_steps_savers = sub_steps_savers

        # Third, strip the sub steps from truncable steps before saving
        del step.steps
        del step.steps_as_tuple

        return step

    def load_step(self, step: 'TruncableSteps', context: ExecutionContext) -> 'TruncableSteps':
        """
        #. Loop through all of the sub steps savers, and only load the sub steps that have been saved.
        #. Refresh steps

        :param step: step to load
        :type step: BaseStep
        :param context: execution context
        :type context: ExecutionContext
        :return: loaded truncable steps
        :rtype: TruncableSteps
        """
        step.steps_as_tuple = []

        for step_name, savers in step.sub_steps_savers:
            if savers is None:
                # keep step as it is if it hasn't been saved
                step.steps_as_tuple.append((step_name, step[step_name]))
            else:
                # Load each sub step with their savers
                sub_step_to_load = Identity(name=step_name, savers=savers)
                sub_step = sub_step_to_load.load(context)
                step.steps_as_tuple.append((step_name, sub_step))

        step._refresh_steps()

        return step

    def get_saved_files(self, step: 'BaseStep', context: 'ExecutionContext') -> List[Tuple[str, str]]:
        saved_files = []
        for step_name, savers in step.sub_steps_savers:
            if savers is None:
                savers = []

            step_identity = Identity(name=step_name, savers=savers)
            for saver in savers:
                saved_files.extend(saver.get_saved_files(step_identity, context.push(step_identity)))

            saved_files.extend(context.stripped_saver.get_saved_files(step_identity, context.push(step_identity)))

        return saved_files


class TruncableSteps(BaseStep, ABC):
    """
    Step that contains multiple steps. :class:`Pipeline` inherits form this class.
    It is possible to truncate this step * :func:`~neuraxle.base.TruncableSteps.__getitem__`

    * self.steps contains the actual steps
    * self.steps_as_tuple contains a list of tuple of step name, and step

    .. seealso::
        :class:`Pipeline`,
        :class:`FeatureUnion`
    """

    def __init__(
            self,
            steps_as_tuple: NamedTupleList,
            hyperparams: HyperparameterSamples = dict(),
            hyperparams_space: HyperparameterSpace = dict()
    ):
        BaseStep.__init__(self, hyperparams=hyperparams, hyperparams_space=hyperparams_space)
        self.set_steps(steps_as_tuple)

        self.set_savers([TruncableJoblibStepSaver()] + self.savers)

    def are_steps_before_index_the_same(self, other: 'TruncableSteps', index: int) -> bool:
        """
        Returns true if self.steps before index are the same as other.steps before index.

        :param other: other truncable steps to compare
        :type other: TruncableSteps
        :param index: max step index to compare
        :type index: int

        :return: bool
        :rtype: bool
        """
        steps_before_index = self[:index]
        for current_index, (step_name, step) in enumerate(steps_before_index):
            source_current_step = inspect.getsource(step.__class__)
            source_cached_step = inspect.getsource(other[current_index].__class__)

            if source_current_step != source_cached_step:
                return False

        return True

    def _load_saved_pipeline_steps_before_index(self, saved_pipeline: 'TruncableSteps', index: int):
        """
        Load the cached pipeline steps
        before the index into the current steps

        :param saved_pipeline: saved pipeline
        :type saved_pipeline: TruncableSteps
        :param index: step index
        :type index: int
        :return:
        """
        self.set_hyperparams(saved_pipeline.get_hyperparams())
        self.set_hyperparams_space(saved_pipeline.get_hyperparams_space())

        new_truncable_steps = saved_pipeline[:index] + self[index:]
        self.set_steps(new_truncable_steps.steps_as_tuple)

    def set_steps(self, steps_as_tuple: NamedTupleList):
        """
        Set steps as tuple.

        :param steps_as_tuple: list of tuple containing step name and step
        :type steps_as_tuple: NamedTupleList
        :return:
        """
        steps_as_tuple = self._wrap_non_base_steps(steps_as_tuple)
        self.steps_as_tuple: NamedTupleList = self._patch_missing_names(steps_as_tuple)
        self._refresh_steps()

    def setup(self) -> 'BaseStep':
        """
        Initialize step before it runs.

        :return: self
        :rtype: BaseStep
        """
        if self.is_initialized:
            return self

        self.is_initialized = True

        return self

    def teardown(self) -> 'BaseStep':
        """
        Teardown step after program execution.
        Teardowns all of the sub steps as well.

        :return: self
        :rtype: BaseStep
        """
        for step_name, step in self.steps_as_tuple:
            step.teardown()

        return self

    def apply(self, method_name: str, *kargs, **kwargs) -> 'BaseStep':
        """
        Apply the method name to the pipeline step and all of its children.

        :param method_name: method name that need to be called on all steps
        :param kargs: any additional arguments to be passed to the method
        :param kwargs: any additional positional arguments to be passed to the method
        :return: self (not a new step)
        :rtype: BaseStep
        """
        BaseStep.apply(self, method_name, *kargs, **kwargs)
        for step in self.values():
            step.apply(method_name, *kargs, **kwargs)
        return self

    def apply_method(self, method: Callable, *kargs, **kwargs) -> 'BaseStep':
        """
        Apply a method to the pipeline step and all of its children.

        :param method: method to call with self
        :param kargs: any additional arguments to be passed to the method
        :param kwargs: any additional positional arguments to be passed to the method
        :return: self (not a new step)
        :rtype: BaseStep
        """
        BaseStep.apply_method(self, method, *kargs, **kwargs)
        for step in self.values():
            step.apply_method(method, *kargs, **kwargs)

        return self

    def get_step_by_name(self, name):
        for step in self.values():
            if step.name == name:
                return step

            found_step = step.get_step_by_name(name)
            if found_step is not None:
                return found_step

        return None

    def _wrap_non_base_steps(self, steps_as_tuple: List) -> NamedTupleList:
        """
        If some steps are not of type BaseStep, we'll try to make them of this type. For instance, sklearn objects
        will be wrapped by a SKLearnWrapper here.

        :param steps_as_tuple: a list of steps or of named tuples of steps (e.g.: NamedTupleList)
        :return: a NamedTupleList
        """
        # TODO: document more the type of the `steps as tuple`.

        wrapped = []
        for step in steps_as_tuple:
            class_name = None
            if isinstance(step, tuple):
                class_name = step[0]
                step = step[1]

            if isinstance(step, BaseEstimator):
                import neuraxle.steps.sklearn
                step = neuraxle.steps.sklearn.SKLearnWrapper(step)
                step.set_name(step.get_wrapped_sklearn_predictor().__class__.__name__)

            if class_name is None:
                class_name = step.get_name()

            wrapped.append((class_name, step))
        return wrapped

    def _patch_missing_names(self, steps_as_tuple: NamedTupleList) -> NamedTupleList:
        """
        Make sure that each sub step has a unique name, and add a name to the sub steps that don't have one already.

        :param steps_as_tuple: a NamedTupleList
        :type steps_as_tuple: a NamedTupleList
        :return: a NamedTupleList with fixed names
        """
        # TODO: document more the type of the `steps as tuple`.
        names_yet = set()
        patched = []
        for class_name, step in steps_as_tuple:
            _name = class_name
            if class_name in names_yet:
                warnings.warn(
                    "Named pipeline tuples must be unique. "
                    "Will rename '{}' because it already exists.".format(class_name))

                _name = self._rename_step(step_name=_name, class_name=class_name, names_yet=names_yet)
                step.set_name(_name)

            step = (_name, step)
            names_yet.add(step[0])
            patched.append(step)
        self.is_invalidated = True
        return patched

    def _rename_step(self, step_name, class_name, names_yet: set):
        """
        Rename step by adding a number suffix after the class name.
        Ensure uniqueness with the names yet parameter.

        :param step_name: step name
        :type step_name: str
        :param class_name: class name
        :type class_name: str
        :param names_yet: names already taken
        :type names_yet: set
        :return: new step name
        :rtype: str
        """
        # Add suffix number to name if it is already used to ensure name uniqueness.
        i = 1
        while step_name in names_yet:
            step_name = class_name + str(i)
            i += 1
        self.is_invalidated = True
        return step_name

    def _refresh_steps(self):
        """
        Private method to refresh inner state after having edited ``self.steps_as_tuple``
        (recreate ``self.steps`` from ``self.steps_as_tuple``).
        """
        self.is_invalidated = True
        self.steps: OrderedDict = OrderedDict(self.steps_as_tuple)
        for name, step in self.items():
            step.name = name

    def get_hyperparams(self) -> HyperparameterSamples:
        """
        Get step hyperparameters as :class:`HyperparameterSamples`.

        Example :

        .. code-block:: python

            p = Pipeline([SomeStep()])
            p.set_hyperparams(HyperparameterSamples({
                'learning_rate': 0.1,
                'some_step__learning_rate': 0.2 # will set SomeStep() hyperparam 'learning_rate' to 0.2
            }))

            hp = p.get_hyperparams()
            # hp ==>  { 'learning_rate': 0.1, 'some_step__learning_rate': 0.2 }

        :return: step hyperparameters
        :rtype: HyperparameterSamples

        .. seealso::
            :class:`HyperparameterSamples`
        """
        hyperparams = dict()

        for k, v in self.steps.items():
            hparams = v.get_hyperparams()  # TODO: oop diamond problem?
            if hasattr(v, "hyperparams"):
                hparams.update(v.hyperparams)
            if len(hparams) > 0:
                hyperparams[k] = hparams

        hyperparams = HyperparameterSamples(hyperparams)

        hyperparams.update(
            BaseStep.get_hyperparams(self)
        )

        return hyperparams.to_flat()

    def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, OrderedDict, dict]) -> BaseStep:
        """
        Set step hyperparameters to the given :class:`HyperparameterSamples`.

        Example :

        .. code-block:: python

            p = Pipeline([SomeStep()])
            p.set_hyperparams(HyperparameterSamples({
                'learning_rate': 0.1,
                'some_step__learning_rate': 0.2 # will set SomeStep() hyperparam 'learning_rate' to 0.2
            }))

        :return: step hyperparameters
        :rtype: HyperparameterSamples

        .. seealso::
            :class:`HyperparameterSamples`
        """
        self.is_invalidated = True

        hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams.items():
            if name in self.steps.keys():
                self.steps[name].set_hyperparams(HyperparameterSamples(hparams))
            else:
                remainders[name] = hparams
        self.hyperparams = HyperparameterSamples(remainders)

        return self

    def update_hyperparams(self, hyperparams: Union[HyperparameterSamples, OrderedDict, dict]) -> BaseStep:
        """
        Update the steps hyperparameters without removing the already-set hyperparameters.
        Please refer to :func:`~BaseStep.update_hyperparams`.

        :param hyperparams: hyperparams to update
        :type hyperparams: HyperparameterSamples
        :return: step
        :rtype: BaseStep

        .. seealso::
            :func:`~BaseStep.update_hyperparams`,
            :class:`HyperparameterSamples`
        """
        self.is_invalidated = True

        hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams.items():
            if name in self.steps.keys():
                self.steps[name].update_hyperparams(HyperparameterSamples(hparams))
            else:
                remainders[name] = hparams
        self.hyperparams.update(remainders)

        return self

    def get_hyperparams_space(self):
        """
        Get step hyperparameters space as :class:`HyperparameterSpace`.

        Example :

        .. code-block:: python

            p = Pipeline([SomeStep()])
            p.set_hyperparams_space(HyperparameterSpace({
                'learning_rate': RandInt(0,5),
                'some_step__learning_rate': RandInt(0, 10) # will set SomeStep() 'learning_rate' hyperparam space to RandInt(0, 10)
            }))

            hp = p.get_hyperparams_space()
            # hp ==>  { 'learning_rate': RandInt(0,5), 'some_step__learning_rate': RandInt(0,10) }

        :return: step hyperparameters space
        :rtype: HyperparameterSpace

        .. seealso::
            :class:`HyperparameterSpace`
        """
        all_hyperparams = HyperparameterSpace()
        for step_name, step in self.steps_as_tuple:
            hspace = step.get_hyperparams_space()
            all_hyperparams.update({
                step_name: hspace
            })
        all_hyperparams.update(
            BaseStep.get_hyperparams_space(self)
        )

        return all_hyperparams.to_flat()

    def set_hyperparams_space(self, hyperparams_space: Union[HyperparameterSpace, OrderedDict, dict]) -> BaseStep:
        """
        Set step hyperparameters space as :class:`HyperparameterSpace`.

        Example :

        .. code-block:: python

            p = Pipeline([SomeStep()])
            p.set_hyperparams_space(HyperparameterSpace({
                'learning_rate': RandInt(0,5),
                'some_step__learning_rate': RandInt(0, 10) # will set SomeStep() 'learning_rate' hyperparam space to RandInt(0, 10)
            }))

        :param hyperparams_space: hyperparameters space
        :type hyperparams_space: Union[HyperparameterSpace, OrderedDict, dict]
        :return: self
        :rtype: BaseStep

        .. seealso::
            :class:`HyperparameterSpace`
        """
        self.is_invalidated = True

        hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams_space.items():
            if name in self.keys():
                self.steps[name].set_hyperparams_space(HyperparameterSpace(hparams))
            else:
                remainders[name] = hparams
        self.hyperparams_space = HyperparameterSpace(remainders)

        return self

    def should_save(self):
        """
        Returns if the step needs to be saved or not.
        If self should be saved or any of his sub steps, return True.

        :return:
        .. seealso::
            :class:`TruncableJoblibStepSaver`
        """
        if BaseStep.should_save(self):
            return True

        for _, step in self.items():
            if step.should_save():
                return True
        return False

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=True) -> 'BaseStep':
        """
        Call mutate on every steps the the present truncable step contains.

        :param new_method: the method to replace transform with.
        :param method_to_assign_to: the method to which the new method will be assigned to.
        :param warn: (verbose) wheter or not to warn about the inexistence of the method.
        :return: self, a copy of self, or even perhaps a new or different BaseStep object.

        .. seealso::
            :func:`~neuraxle.base.BaseStep.reverse`
            :func:`~neuraxle.base.BaseStep.inverse_transform`
        """
        if self.pending_mutate[0] is None:
            new_base_step = self
            self.pending_mutate = (new_base_step, self.pending_mutate[1], self.pending_mutate[2])

            new_base_step.steps_as_tuple = [
                (
                    k,
                    v.mutate(new_method, method_to_assign_to, warn)
                )
                for k, v in new_base_step.steps_as_tuple
            ]
            new_base_step._refresh_steps()
            return BaseStep.mutate(self, new_method, method_to_assign_to, warn)
        else:
            return BaseStep.mutate(self, new_method, method_to_assign_to, warn)

    def _step_index_to_name(self, step_index):
        if step_index == len(self.items()):
            return None

        name, _ = self.steps_as_tuple[step_index]
        return name

    def __setitem__(self, key: Union[slice, int, str], new_step: BaseStep):
        """
        Set one step with a key, and a value.

        :param key: slice, index, or step name
        :type key: Union[slice, int, str]
        :param new_step: step
        :type new_step: BaseStep
        """
        if isinstance(key, str):
            index = 0
            for step_index, (current_step_name, step) in enumerate(self.steps_as_tuple):
                if current_step_name == key:
                    index = step_index

            new_step.set_name(key)
            self.steps[index] = new_step
            self.steps_as_tuple[index] = (key, new_step)
        else:
            raise ValueError(
                'type {0} not supported yet in TruncableSteps.__setitem__, please implement it if you need it'.format(
                    type(key)))

    def __getitem__(self, key: Union[slice, int, str]):
        """
        Truncate self with a slice, an index or a step name.

        Example :

        .. code-block:: python

            p = Pipeline([
                ('1', SomeStep()),
                ('2', SomeStep()),
                ('3', SomeStep())
            ])
            p[0] # returns the first SomeStep()
            p[0:2] # returns a TruncableSteps containing the first, and second SomeStep()
            p['2'] # returns the second SomeStep()

        :param key: slice, index, or step name
        :type key: Union[slice, int, str]

        :return: truncated self
        :rtype: Union[TruncableSteps, BaseStep]


        .. seealso::
            :class:`DataContainer`,
            `Getting model attributes from scikit-learn pipeline on stackoverflow <https://stackoverflow.com/questions/28822756/getting-model-attributes-from-scikit-learn-pipeline/58359509#58359509>`_
        """
        if isinstance(key, slice):
            self_shallow_copy = copy(self)

            if isinstance(key.start, int):
                start = self._step_index_to_name(key.start)
            else:
                start = key.start

            if isinstance(key.stop, int):
                stop = self._step_index_to_name(key.stop)
            else:
                stop = key.stop

            step = key.step
            if step is not None or (start is None and stop is None):
                raise KeyError("Invalid range: '{}'.".format(key))
            new_steps_as_tuple = []
            if start is None:
                if stop not in self.steps.keys():
                    raise KeyError("Stop '{}' not found in '{}'.".format(stop, self.steps.keys()))
                for key, val in self.steps_as_tuple:
                    if stop == key:
                        break
                    new_steps_as_tuple.append((key, val))
            elif stop is None:
                if start not in self.steps.keys():
                    raise KeyError("Start '{}' not found in '{}'.".format(stop, self.steps.keys()))
                for key, val in reversed(self.steps_as_tuple):
                    new_steps_as_tuple.append((key, val))
                    if start == key:
                        break
                new_steps_as_tuple = list(reversed(new_steps_as_tuple))
            else:
                started = False
                if stop not in self.steps.keys() or start not in self.steps.keys():
                    raise KeyError(
                        "Start or stop ('{}' or '{}') not found in '{}'.".format(start, stop, self.steps.keys()))
                for key, val in self.steps_as_tuple:
                    if start == stop == key:
                        new_steps_as_tuple.append((key, val))

                    if stop == key:
                        break

                    if not started and start == key:
                        started = True
                    if started:
                        new_steps_as_tuple.append((key, val))

            self_shallow_copy.steps_as_tuple = new_steps_as_tuple
            self_shallow_copy.steps = OrderedDict(new_steps_as_tuple)
            return self_shallow_copy
        else:
            if isinstance(key, int):
                key = self._step_index_to_name(key)

            return self.steps[key]

    def __add__(self, other: 'TruncableSteps') -> 'TruncableSteps':
        """
        Concatenate the given truncable steps to self.

        :param other: other truncable steps
        :type other: TruncableSteps
        :return: new truncable steps with concatenated steps
        :rtype: TruncableSteps
        """
        self.set_steps(self.steps_as_tuple + other.steps_as_tuple)
        return self

    def items(self) -> ItemsView:
        """
        Returns all of the steps as tuples items (step_name, step).

        :return: step items tuple : (step name, step)
        :rtype: ItemsView
        """
        return self.steps.items()

    def keys(self) -> KeysView:
        """
        Returns the step names.

        :return: list of step names
        :rtype: KeysView
        """
        return self.steps.keys()

    def values(self) -> ValuesView:
        """
        Get step values.

        :return: all of the steps
        :rtype: ValuesView
        """
        return self.steps.values()

    def append(self, item: Tuple[str, 'BaseStep']) -> 'TruncableSteps':
        """
        Add an item to steps as tuple.

        :param item: item tuple (step name, step)
        :type item: Tuple[str, 'BaseStep']
        :return: self
        :rtype: TruncableSteps
        """
        self.steps_as_tuple.append(item)
        self._refresh_steps()
        return self

    def pop(self) -> 'BaseStep':
        """
        Pop the last step.

        :return: last step
        :rtype: BaseStep
        """
        return self.popitem()[-1]

    def popitem(self, key=None) -> Tuple[str, 'BaseStep']:
        """
        Pop the last step, or the step with the given key

        :param key: step name to pop, or None
        :type key: str
        :return: last step item
        :rtype: Tuple[str, BaseStep]

        """
        if key is None:
            item = self.steps_as_tuple.pop()
            self._refresh_steps()
        else:
            item = key, self.steps.pop(key)
            self.steps_as_tuple = list(self.steps.items())
        return item

    def popfront(self) -> 'BaseStep':
        """
        Pop the first step.

        :return: first step
        :rtype: BaseStep
        """
        return self.popfrontitem()[-1]

    def popfrontitem(self) -> Tuple[str, 'BaseStep']:
        """
        Pop the first step.

        :return: first step item
        :rtype: Tuple[str, BaseStep]
        """
        item = self.steps_as_tuple.pop(0)
        self._refresh_steps()
        return item

    def __contains__(self, item):
        """
        Check wheter the ``item`` key or value (or key value tuple pair) is found in self.

        :param item: The key or value to check if is in self's keys or values.
        :return: True or False
        """
        return item in self.steps.keys() or item in self.steps.values() or item in self.items()

    def __iter__(self):
        """
        Iterate through the steps.

        :return: iter(self.steps_as_tuple)
        """
        return iter(self.steps_as_tuple)

    def __len__(self):
        """
        Return the number of contained steps.

        :return: len(self.steps_as_tuple)
        """
        return len(self.steps_as_tuple)

    def split(self, step_type: type) -> List['TruncableSteps']:
        """
        Split truncable steps by a step class name.

        :param step_type: step class type to split from.
        :type step_type: str
        :return: list of truncable steps containing the splitted steps
        """
        sub_pipelines = []

        previous_sub_pipeline_end_index = 0
        for index, (step_name, step) in enumerate(self.items()):
            if isinstance(step, step_type):
                sub_pipelines.append(
                    self[previous_sub_pipeline_end_index:index + 1]
                )
                previous_sub_pipeline_end_index = index + 1

        if previous_sub_pipeline_end_index < len(self.items()):
            sub_pipelines.append(
                self[previous_sub_pipeline_end_index:-1]
            )

        return sub_pipelines

    def ends_with(self, step_type: type):
        """
        Returns true if truncable steps end with a step of the given type.

        :param step_type: step type
        :type step_type: type

        :return: if truncable steps ends with the given step type
        :rtype: bool
        """
        return isinstance(self[-1], step_type)

    def set_train(self, is_train: bool = True) -> 'BaseStep':
        """
        Set pipeline step mode to train or test.

        In the pipeline steps functions, you can add a simple if statement to direct to the right implementation:

        .. code-block:: python

            def transform(self, data_inputs):
                if self.is_train:
                    self.transform_train_(data_inputs)
                else:
                    self.transform_test_(data_inputs)

            def fit_transform(self, data_inputs, expected_outputs):
                if self.is_train:
                    self.fit_transform_train_(data_inputs, expected_outputs)
                else:
                    self.fit_transform_test_(data_inputs, expected_outputs)

        :param is_train: if the step is in train mode (True) or test mode (False)
        :type is_train: bool
        :return: self
        """
        self.is_train = is_train
        for _, step in self.items():
            step.set_train(is_train)
        return self

    def __repr__(self):

        output = self.__class__.__name__ + '\n' \
                 + "(\n\t" + super(TruncableSteps, self).__repr__() \
                 + "(\n\t\t" + pprint.pformat(self.steps_as_tuple) \
                 + "\t\n)" \
                 + "\n)"

        return output

    def __str__(self):
        return self.__repr__()


class ResumableStepMixin:
    """
    Mixin to add resumable function to a step, or a class that can be resumed, for example a checkpoint on disk.
    """

    @abstractmethod
    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        """
        Returns True if a step can be resumed with the given the data container, and execution context.
        See Checkpoint class documentation for more details on how a resumable checkpoint works.

        :param data_container: data container to resume from
        :param context: execution context to resume from
        :return: if we can resume
        :rtype: bool
        """
        raise NotImplementedError()

    def __str__(self):
        return self.__repr__()


class Identity(NonTransformableMixin, NonFittableMixin, BaseStep):
    """
    A pipeline step that has no effect at all but to return the same data without changes.

    This can be useful to concatenate new features to existing features, such as what AddFeatures do.

    Identity inherits from :class:`NonTransformableMixin` and from :class:`NonFittableMixin` which makes it a class that has no
    effect in the pipeline: it doesn't require fitting, and at transform-time, it returns the same data it received.

    .. seealso::
        :class:`NonTransformableMixin`,
        :class:`NonFittableMixin`,
        :class:`BaseStep`
    """

    def __init__(self, savers=None, name=None):
        if savers is None:
            savers = [JoblibStepSaver()]
        NonTransformableMixin.__init__(self)
        NonFittableMixin.__init__(self)
        BaseStep.__init__(self, name=name, savers=savers)


class FullDumpLoader(Identity):
    """
    Identity step that can load the full dump of a pipeline step.
    Used by :func:`~neuraxle.base.BaseStep.load`.

    .. seealso::
        :class:`ExecutionContext`
        :class:`BaseStep`,
        :class:`Identity`
    """
    def __init__(self, name, stripped_saver=None):
        if stripped_saver is None:
            stripped_saver = JoblibStepSaver()
        Identity.__init__(self, name=name, savers=[stripped_saver])

    def load(self, context: ExecutionContext, full_dump=True) -> BaseStep:
        """
        Load the full dump of a pipeline step.

        :param context: execution context
        :param full_dump: load full dump or not (always true, inherited from :class:`BaseStep`
        :type full_dump: load full dump or not
        :return: loaded step
        :rtype: BaseStep
        """
        if not context.stripped_saver.can_load(self, context):
            path = os.path.join(context.get_path(), self.name)
            raise Exception('Cannot Load Full Dump For Step {}'.format(path))

        loaded_self = context.stripped_saver.load_step(self, context)

        context.pop()
        return loaded_self.load(context, full_dump)
