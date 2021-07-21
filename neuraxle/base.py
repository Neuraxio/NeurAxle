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
import logging
import os
import pprint
import traceback
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy
from enum import Enum
from typing import List, Union, Any, Iterable, KeysView, ItemsView, ValuesView, Callable, Dict, Tuple, Type

from joblib import dump, load

from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples, RecursiveDict

DEFAULT_CACHE_FOLDER = os.path.join(os.getcwd(), 'cache')

LOGGER_FORMAT = "[%(asctime)s][%(levelname)s][%(module)s][%(lineno)d]: %(message)s"
DATE_FORMAT = "%H:%M:%S"
logging.basicConfig(format=LOGGER_FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)

class BaseHasher(ABC):
    """
    Base class to hash hyperparamters, and data input ids together.
    The :class:`~neuraxle.data_container.DataContainer` class uses the hashed values for its current ids.
    :class:`BaseStep` uses many :class:`BaseHasher` objects
    to hash hyperparameters, and data inputs ids together after each transform.

    .. seealso::
        :class:`~neuraxle.data_container.DataContainer`

    """

    @abstractmethod
    def single_hash(self, current_id: str, hyperparameters: HyperparameterSamples) -> List[str]:
        """
        Hash summary id, and hyperparameters together.

        :param current_id: current hashed id
        :param hyperparameters: step hyperparameters to hash with current ids
        :type hyperparameters: HyperparameterSamples
        :return: the new hashed current id
        """
        raise NotImplementedError()

    @abstractmethod
    def hash(self, current_ids: List[str], hyperparameters: HyperparameterSamples, data_inputs: Iterable) -> List[str]:
        """
        Hash :class:`~neuraxle.data_container.DataContainer`.current_ids, data inputs, and hyperparameters together.

        :param current_ids: current hashed ids (can be None if this function has not been called yet)
        :param hyperparameters: step hyperparameters to hash with current ids
        :param data_inputs: data inputs to hash current ids for
        :return: the new hashed current ids
        """
        raise NotImplementedError()


class HashlibMd5Hasher(BaseHasher):
    """
    Class to hash hyperparamters, and data input ids together using md5 algorithm from hashlib :
    `<https://docs.python.org/3/library/hashlib.html>`_

    The :class:`~neuraxle.data_container.DataContainer` class uses the hashed values for its current ids.
    :class:`BaseStep` uses many :class:`BaseHasher` objects
    to hash hyperparameters, and data inputs ids together after each transform.

    .. seealso::
        :class:`~neuraxle.base.BaseHasher`,
        :class:`~neuraxle.data_container.DataContainer`

    """

    def single_hash(self, current_id: str, hyperparameters: HyperparameterSamples) -> List[str]:
        """
        Hash summary id, and hyperparameters together.

        :param current_id: current hashed id
        :param hyperparameters: step hyperparameters to hash with current ids
        :return: the new hashed current id
        """
        m = hashlib.md5()

        current_hyperparameters_hash = hashlib.md5(
            str.encode(str(hyperparameters.to_flat_dict()))
        ).hexdigest()

        m.update(str.encode(str(current_id)))
        m.update(str.encode(current_hyperparameters_hash))

        return m.hexdigest()

    def hash(self, current_ids, hyperparameters, data_inputs: Any = None) -> List[str]:
        """
        Hash :class:`~neuraxle.data_container.DataContainer`.current_ids, data inputs, and hyperparameters together
        using  `hashlib.md5 <https://docs.python.org/3/library/hashlib.html>`_

        :param current_ids: current hashed ids (can be None if this function has not been called yet)
        :param hyperparameters: step hyperparameters to hash with current ids
        :param data_inputs: data inputs to hash current ids for
        :return: the new hashed current ids
        """
        if current_ids is None:
            if isinstance(data_inputs, Iterable):
                current_ids = [str(i) for i in range(len(data_inputs))]
            else:
                current_ids = [str(0)]

        if len(hyperparameters) == 0:
            return current_ids

        hyperperams_dict = hyperparameters.to_flat_dict()
        current_hyperparameters_hash = hashlib.md5(str.encode(str(hyperperams_dict))).hexdigest()

        new_current_ids = []
        for current_id in current_ids:
            m = hashlib.md5()
            m.update(str.encode(current_id))
            m.update(str.encode(current_hyperparameters_hash))
            new_current_ids.append(m.hexdigest())

        return new_current_ids


class HashlibMd5ValueHasher(HashlibMd5Hasher):
    def hash(self, current_ids, hyperparameters, data_inputs: Any = None) -> List[str]:
        """
        Hash :class:`~neuraxle.data_container.DataContainer`.current_ids, data inputs, and hyperparameters together
        using  `hashlib.md5 <https://docs.python.org/3/library/hashlib.html>`_

        :param current_ids: current hashed ids (can be None if this function has not been called yet)
        :param hyperparameters: step hyperparameters to hash with current ids
        :param data_inputs: data inputs to hash current ids for
        :return: the new hashed current ids
        """
        if len(current_ids) != len(data_inputs):
            current_ids: List[str] = [str(i) for i in range(len(data_inputs))]

        if current_ids is None:
            if isinstance(data_inputs, Iterable):
                current_ids = [str(i) for i in range(len(data_inputs))]
            else:
                current_ids = [str(0)]

        if len(hyperparameters) == 0:
            return current_ids

        hyperperams_dict = hyperparameters.to_flat_dict()
        current_hyperparameters_hash = hashlib.md5(str.encode(str(hyperperams_dict))).hexdigest()

        new_current_ids = []
        for current_id, di in zip(current_ids, data_inputs):
            m = hashlib.md5()
            m.update(str.encode(current_id))
            m.update(str.encode(str(di)))
            m.update(str.encode(current_hyperparameters_hash))
            new_current_ids.append(m.hexdigest())

        return new_current_ids


class BaseSaver(ABC):
    """
    Any saver must inherit from this one. Some savers just save parts of objects, some save it all or what remains.
    Each :class`BaseStep` can potentially have multiple savers to make serialization possible.

    .. seealso::
        :func:`~neuraxle.base._HasSavers.save`,
        :func:`~neuraxle.base._HasSavers.load`
    """

    @abstractmethod
    def save_step(self, step: 'BaseTransformer', context: 'ExecutionContext') -> 'BaseTransformer':
        """
        Save step with execution context.

        :param step: step to save
        :param context: execution context
        :param save_savers:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def can_load(self, step: 'BaseTransformer', context: 'ExecutionContext'):
        """
        Returns true if we can load the given step with the given execution context.

        :param step: step to load
        :param context: execution context to load from
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def load_step(self, step: 'BaseTransformer', context: 'ExecutionContext') -> 'BaseTransformer':
        """
        Load step with execution context.

        :param step: step to load
        :param context: execution context to load from
        :return: loaded base step
        """
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
        :class:`~neuraxle.base.BaseSaver`,
        :class:`~neuraxle.base.ExecutionContext`
    """

    def can_load(self, step: 'BaseTransformer', context: 'ExecutionContext') -> bool:
        """
        Returns true if the given step has been saved with the given execution context.

        :param step: step that might have been saved
        :param context: execution context
        :return: if we can load the step with the given context
        """
        return os.path.exists(
            self._create_step_path(context, step)
        )

    def _create_step_path(self, context, step):
        """
        Create step path for the given context.

        :param context: execution context
        :param step: step to save, or load
        :return: path
        """
        return os.path.join(context.get_path(), '{0}.joblib'.format(step.name))

    def save_step(self, step: 'BaseTransformer', context: 'ExecutionContext') -> 'BaseTransformer':
        """
        Saved step stripped out of things that would make it unserializable.

        :param step: stripped step to save
        :param context: execution context to save from
        :return:
        """
        context.mkdir()

        path = self._create_step_path(context, step)
        dump(step, path)

        return step

    def load_step(self, step: 'BaseTransformer', context: 'ExecutionContext') -> 'BaseTransformer':
        """
        Load stripped step.

        :param step: stripped step to load
        :param context: execution context to load from
        :return:
        """
        loaded_step = load(self._create_step_path(context, step))

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
    INVERSE_TRANSFORM = 'inverse_transform'


class ExecutionPhase(Enum):
    UNSPECIFIED = None
    PRETRAIN = "pretraining"
    TRAIN = "training"
    VALIDATION = "validation"
    TEST = "test"
    PROD = "production"


class ExecutionContext:
    """
    Execution context object containing all of the pipeline hierarchy steps.
    First item in execution context parents is root, second is nested, and so on. This is like a stack.

    The execution context is used for fitted step saving, and caching :
        * :func:`~neuraxle.base._HasSavers.save`
        * :func:`~neuraxle.base._HasSavers.load`
        * :func:`~neuraxle.steps.caching.ValueCachingWrapper.handle_transform`
        * :func:`~neuraxle.steps.caching.ValueCachingWrapper.handle_fit_transform`

    .. seealso::
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.steps.caching.ValueCachingWrapper`
    """

    def __init__(
            self,
            root: str = DEFAULT_CACHE_FOLDER,
            execution_phase: ExecutionPhase = ExecutionPhase.UNSPECIFIED,
            execution_mode: ExecutionMode = ExecutionMode.FIT_OR_FIT_TRANSFORM_OR_TRANSFORM,
            stripped_saver: BaseSaver = None,
            parents: List['BaseStep'] = None,
            services: Dict[Type, object] = None,
            logger: logging.Logger = None
    ):

        self.execution_mode = execution_mode
        self.execution_phase = execution_phase

        if stripped_saver is None:
            stripped_saver: BaseSaver = JoblibStepSaver()

        self.stripped_saver = stripped_saver
        self.root: str = root
        if parents is None:
            parents = []
        self.parents: List[BaseStep] = parents

        if services is None:
            services: Dict[Type, object] = dict()
        self.services: Dict[Type, object] = services

        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

    def set_execution_phase(self, phase: ExecutionPhase) -> 'ExecutionContext':
        """
        Set the instance's execution phase to given phase.

        :param phase:
        :return:
        """
        self.execution_phase: ExecutionPhase = phase
        return self

    def set_service_locator(self, services: Dict[Type, object]) -> 'ExecutionContext':
        """
        Register abstract class type instances.

        :param services:  instance
        :return: self
        """
        self.services: Dict[Type, object] = services
        return self

    def register_service(self, service_abstract_class_type: Type, service_instance: object) -> 'ExecutionContext':
        """
        Register base class instance inside the services.

        :param service_abstract_class_type: base type
        :param service_instance:  instance
        :return: self
        """
        self.services[service_abstract_class_type] = service_instance
        return self

    def get_service(self, service_abstract_class_type: Type) -> object:
        """
        Get the registered instance for the given abstract class type.

        :param service_abstract_class_type: base type
        :return: self
        """
        return self.services[service_abstract_class_type]

    def get_services(self) -> object:
        """
        Get the registered instances in the services.

        :return: self
        """
        return self.services

    def has_service(self, service_abstract_class_type: Type) -> bool:
        """
        Return a bool indicating if the service has been registered.

        :param service_abstract_class_type: base type
        :return: if the service registered or not
        """
        return service_abstract_class_type in self.services

    def set_logger(self, logger):
        self.logger = logger
        return self

    def get_execution_mode(self) -> ExecutionMode:
        return self.execution_mode

    def save(self, full_dump=False):
        """
        Save all unsaved steps in the parents of the execution context using :func:`~neuraxle.base._HasSavers.save`.
        This method is called from a step checkpointer inside a :class:`Checkpoint`.

        :param full_dump: save full pipeline dump to be able to load everything without source code (false by default).
        :return:

        .. seealso::
            :class:`BaseStep`,
            :func:`~neuraxle.base._HasSavers.save`
        """
        while not self.empty():
            should_save_last_step = self.should_save_last_step()
            last_step = self.peek()
            if full_dump:
                should_save_last_step = True

            self.pop()
            if should_save_last_step:
                last_step.save(self, full_dump)

    def save_last(self):
        """
        Save only the last step in the execution context.

        .. seealso::
            :func:`~neuraxle.base.ExecutionContext.save`
        """
        last_step = self.peek()
        self.pop()
        last_step.save(self, True)

    def should_save_last_step(self) -> bool:
        """
        Returns True if the last step should be saved.

        :return: if the last step should be saved
        """
        if len(self.parents) > 0:
            return self.parents[-1].should_save()
        return False

    def pop_item(self) -> 'BaseTransformer':
        """
        Change the execution context to be the same as the latest parent context.

        :return:
        """
        return self.parents.pop()

    def pop(self) -> bool:
        """
        Pop the context. Returns True if it successfully popped an item from the parents list.

        :return: if an item has been popped
        """
        if len(self) == 0:
            return False
        self.pop_item()
        return True

    def push(self, step: 'BaseTransformer') -> 'ExecutionContext':
        """
        Pushes a step in the parents of the execution context.

        :param step: step to add to the execution context
        :return: self
        """
        return ExecutionContext(
            root=self.root,
            execution_mode=self.execution_mode,
            execution_phase=self.execution_phase,
            parents=self.parents + [step],
            services=self.services,
            logger=self.logger
        )

    def copy(self):
        return ExecutionContext(
            root=self.root,
            execution_mode=self.execution_mode,
            execution_phase=self.execution_phase,
            parents=copy(self.parents),
            services=self.services,
            logger=self.logger
        )

    def peek(self) -> 'BaseTransformer':
        """
        Get last parent.

        :return: the last parent base step
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

    def get_path(self, is_absolute: bool = True):
        """
        Creates the directory path for the current execution context.

        :param is_absolute: bool to say if we want to add root to the path or not
        :return: current context path
        """
        parents_with_path = [self.root] if is_absolute else []
        parents_with_path += [p.name for p in self.parents]
        if len(parents_with_path) == 0:
            return '.' + os.sep
        return os.path.join(*parents_with_path)

    def get_names(self):
        """
        Returns a list of the parent names.

        :return: list of parents step names
        """
        return [p.name for p in self.parents]

    def empty(self):
        """
        Return True if the context doesn't have parent steps.

        :return: if parents len is 0
        """
        return len(self) == 0

    def load(self, path: str) -> 'BaseTransformer':
        """
        Load full dump at the given path.

        :param path: pipeline step path
        :return: loaded step

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

    def __len__(self):
        return len(self.parents)


class _RecursiveArguments:
    """
    This class is used by :func:`~neuraxle.base.BaseStep.apply`, and :class:`_HasChildrenMixin` to pass the right arguments to steps with children.

    .. seealso::
        :class:`_HasChildrenMixin`,
        :func:`~neuraxle.base._HasHyperparamsSpace.set_hyperparams_space`,
        :func:`~neuraxle.base._HasHyperparamsSpace.get_hyperparams_space`,
        :func:`~neuraxle.base._HasHyperparams.get_hyperparams`,
        :func:`~neuraxle.base._HasHyperparams.set_hyperparams`,
        :func:`~neuraxle.base._HasHyperparams.update_hyperparams`,
        :func:`~neuraxle.base._HasHyperparamsSpace.update_hyperparams_space`,
        :func:`~neuraxle.base.BaseTransformer.invalidate`
    """

    def __init__(self, ra=None, *args, **kwargs):
        if ra is not None:
            args = ra.args
            kwargs = ra.kwargs
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, child_step_name: str):
        """
        Return recursive arguments for the given child step name.
        If child step name is None, return the root values.

        :param child_step_name: child step name, or None if we want to get root values.
        :return: recursive argument for the given child step name
        """
        if child_step_name is None:
            arguments = list()
            keyword_arguments = dict()
            for arg in self.args:
                if isinstance(arg, RecursiveDict):
                    arguments.append(arg.get(child_step_name))
                else:
                    arguments.append(arg)
            for key, arg in self.kwargs.items():
                if isinstance(arg, RecursiveDict):
                    keyword_arguments[key] = arg.get(child_step_name)
                else:
                    keyword_arguments[key] = arg
            return _RecursiveArguments(*arguments, **keyword_arguments)
        else:
            arguments = list()
            keyword_arguments = dict()
            for arg in self.args:
                if isinstance(arg, RecursiveDict):
                    arguments.append(arg.get(child_step_name))
                else:
                    arguments.append(arg)
            for key, arg in self.kwargs.items():
                if isinstance(arg, RecursiveDict):
                    keyword_arguments[key] = arg.get(child_step_name)
                else:
                    keyword_arguments[key] = arg
            return _RecursiveArguments(*arguments, **keyword_arguments)

    def __iter__(self):
        return self.kwargs


class _HasRecursiveMethods:
    """
    An internal class to represent a step that has recursive methods.
    The apply :func:`apply` function is used to apply a method to a step and its children.


    Example usage :


    .. code-block:: python

        class _HasHyperparams:
            # ...
            def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, Dict]) -> HyperparameterSamples:
                self.apply(method='_set_hyperparams', hyperparams=HyperparameterSamples(hyperparams))
                return self

            def _set_hyperparams(self, hyperparams: Union[HyperparameterSamples, Dict]) -> HyperparameterSamples:
                self._invalidate()
                hyperparams = HyperparameterSamples(hyperparams)
                self.hyperparams = hyperparams if len(hyperparams) > 0 else self.hyperparams
                return self.hyperparams

        pipeline = Pipeline([
            SomeStep()
        ])

        pipeline.set_hyperparams(HyperparameterSamples({
            'learning_rate': 0.1,
            'SomeStep__learning_rate': 0.05
        }))


    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`_HasChildrenMixin`
        :class:`_RecursiveArguments`
    """

    def apply(self, method: Union[str, Callable], ra: _RecursiveArguments = None, *args, **kwargs) -> RecursiveDict:
        """
        Apply a method to a step and its children.

        :param method: method name that need to be called on all steps
        :param ra: recursive arguments
        :param args: any additional arguments to be passed to the method
        :param kwargs: any additional positional arguments to be passed to the method
        :return: method outputs, or None if no method has been applied

        .. seealso::
            :class:`_RecursiveArguments`,
            :class:`_HasChildrenMixin`
        """
        if ra is None:
            ra = _RecursiveArguments(*args, **kwargs)

        kargs = ra.args

        def _return_empty(*args, **kwargs):
            return RecursiveDict()

        _method = _return_empty
        if isinstance(method, str) and hasattr(self, method) and callable(getattr(self, method)):
            _method = getattr(self, method)

        if not isinstance(method, str):
            _method = method
            kargs = [self] + list(kargs)

        try:
            results = _method(*kargs, **ra.kwargs)
            if not isinstance(results, RecursiveDict):
                raise ValueError(
                    'Method {} must return a RecursiveDict because it is applied recursively.'.format(method))
            return results
        except Exception as err:
            print('{}: Failed to apply method {}.'.format(self.name, method))
            print(traceback.format_stack())
            raise err


class _TransformerStep(ABC):
    """
    An internal class to represent a step that can be transformed, or inverse transformed.
    See :class:`BaseTransformer`, for the complete transformer step that can be used inside a :class:`neuraxle.pipeline.Pipeline`.
    See :class:`BaseStep`, for a step that can also be fitted inside a :class:`neuraxle.pipeline.Pipeline`.

    Every step must implement :func:`~neuraxle.base._TransformerStep.transform`.
    If a step is not transformable, you can inherit from :class:`NonTransformableMixin`.

    Every transformer step has handle methods that can be overridden to add side effects or change the execution flow based on the execution context, and the data container :
        * :func:`~neuraxle.base._TransformerStep.handle_transform`
        * :func:`~neuraxle.base._TransformerStep.handle_fit_transform`
        * :func:`~neuraxle.base._TransformerStep.handle_fit`

    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`_FittableStep`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def _will_process(self, data_container: DataContainer, context: ExecutionContext) -> (
            DataContainer, ExecutionContext):
        """
        Apply side effects before any step method.
        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        return data_container, context

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Override this to add side effects or change the execution flow before (or after) calling * :func:`~neuraxle.base._TransformerStep.transform`.
        The default behavior is to rehash current ids with the step hyperparameters.

        :param data_container: the data container to transform
        :param context: execution context
        :return: transformed data container
        """
        if not self.is_initialized:
            self.setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_transform_data_container(data_container, context)

        data_container = self._transform_data_container(data_container, context)

        data_container = self._did_transform(data_container, context)
        data_container = self._did_process(data_container, context)

        return data_container

    def _will_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> \
            (DataContainer, ExecutionContext):
        """
        Apply side effects before transform.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        return data_container, context.push(self)

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Transform data container.

        :param data_container: data container
        :param context: execution context
        :return: data container
        """
        data_container.set_data_inputs(self(data_container.data_inputs))
        return data_container

    def __call__(self, *args, **kwargs) -> Any:
        return self.transform(*args)

    @abstractmethod
    def transform(self, data_inputs):
        """
        Transform given data inputs.

        :param data_inputs: data inputs
        :return: transformed data inputs
        """
        raise NotImplementedError(
            "TODO: Implement this method in {}, or have this class inherit from the NonTransformableMixin.".format(
                self.__class__.__name__))

    def _did_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Apply side effects after transform.

        :param data_container: data container
        :param context: execution context
        :return: data container
        """
        return data_container

    def _did_process(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Apply side effects after any step method.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        data_container = self.hash_data_container(data_container)
        return data_container

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseTransformer':
        """
        Override this to add side effects or change the execution flow before (or after) calling :func:`~neuraxle.base._FittableStep.fit`.
        The default behavior is to rehash current ids with the step hyperparameters.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)

        .. seealso::
            :class:`~neuraxle.data_container.DataContainer`,
            :class:`~neuraxle.pipeline.Pipeline`
        """
        if not self.is_initialized:
            self.setup(context)

        self._did_process(data_container, context)
        return self

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> \
            ('BaseTransformer', DataContainer):
        """
        Override this to add side effects or change the execution flow before (or after) calling * :func:`~neuraxle.base._FittableStep.fit_transform`.
        The default behavior is to rehash current ids with the step hyperparameters.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)
        """
        return self, self.handle_transform(data_container, context)

    def fit(self, data_inputs, expected_outputs) -> '_TransformerStep':
        """
        Fit given data inputs. By default, a step only transforms in the fit transform method.
        To add fitting to your step, see class:`_FittableStep` for more info.
        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on
        :return: transformed data inputs
        """
        return self

    def fit_transform(self, data_inputs, expected_outputs=None):
        """
        Fit transform given data inputs. By default, a step only transforms in the fit transform method.
        To add fitting to your step, see class:`_FittableStep` for more info.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on
        :return: transformed data inputs
        """
        return self, self.transform(data_inputs)

    def handle_predict(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Handle_transform in test mode.

        :param data_container: the data container to transform
        :param context: execution context
        :return: transformed data container
        """
        was_train: bool = self.is_train
        self.set_train(False)

        data_container = self.handle_transform(data_container, context)

        self.set_train(was_train)
        return data_container

    def predict(self, data_input):
        """
        Predict the expected output in test mode using func:`~neuraxle.base._TransformerStep.transform`, but by setting self to test mode first and then reverting the mode.

        :param data_input: data input to predict
        :return: prediction
        """
        was_train: bool = self.is_train
        self.set_train(False)

        outputs = self(data_input)

        self.set_train(was_train)
        return outputs

    def handle_inverse_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Override this to add side effects or change the execution flow before (or after) calling :func:`~neuraxle.base._TransformerStep.inverse_transform`.
        The default behavior is to rehash current ids with the step hyperparameters.

        :param data_container: the data container to inverse transform
        :param context: execution context
        :return: data_container

        .. seealso::
            :class:`~neuraxle.data_container.DataContainer`,
            :class:`~neuraxle.pipeline.Pipeline`
        """
        if not self.is_initialized:
            self.setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container = self._inverse_transform_data_container(data_container, context)
        data_container = self._did_process(data_container, context)

        return data_container

    def _inverse_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> \
            DataContainer:
        processed_outputs = self.inverse_transform(data_container.data_inputs)
        data_container.set_data_inputs(processed_outputs)

        return data_container

    def inverse_transform(self, processed_outputs):
        """
        Inverse Transform the given transformed data inputs.

        :func:`~neuraxle.base.BaseStep.mutate` or :func:`~neuraxle.base.BaseTransformer.reverse` can be called to change the default transform behavior :


        .. code-block:: python

            p = Pipeline([MultiplyBy()])
            _in = np.array([1, 2])
            _out = p.transform(_in)
            _regenerated_in = reversed(p).transform(_out)
            assert np.array_equal(_regenerated_in, _in)


        :param processed_outputs: processed data inputs
        :return: inverse transformed processed outputs
        """
        raise NotImplementedError("TODO: Implement this method in {}.".format(self.__class__.__name__))

    def teardown(self) -> 'BaseTransformer':
        """
        Applies _teardown on the step and, if applicable, its children.
        :return: self
        """
        self.apply("_teardown")
        return self

class _FittableStep:
    """
    An internal class to represent a step that can be fitted.
    See :class:`BaseStep`, for a complete step that can be transformed, and fitted inside a :class:`neuraxle.pipeline.Pipeline`.

    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseStep':
        """
        Override this to add side effects or change the execution flow before (or after) calling :func:`~neuraxle.base._FittableStep.fit`.
        The default behavior is to rehash current ids with the step hyperparameters.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)
        """
        if not self.is_initialized:
            self.setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_fit(data_container, context)

        new_self = self._fit_data_container(data_container, context)

        self._did_fit(data_container, context)
        self._did_process(data_container, context)

        return new_self

    def _will_fit(self, data_container: DataContainer, context: ExecutionContext) -> (DataContainer, ExecutionContext):
        """
        Before fit is called, apply side effects on the step, the data container, or the execution context.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        self._invalidate()
        return data_container, context.push(self)

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> '_FittableStep':
        """
        Fit data container.

        :param data_container: data container
        :param context: execution context
        :return: (fitted self, data container)
        """
        return self.fit(data_container.data_inputs, data_container.expected_outputs)

    def _did_fit(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Apply side effects before fit is called.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        return data_container

    @abstractmethod
    def fit(self, data_inputs, expected_outputs) -> '_FittableStep':
        """
        Fit data inputs on the given expected outputs.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on.
        :return: self
        """
        raise NotImplementedError(
            "TODO: Implement this method in {}, or have this class inherit from the NonFittableMixin.".format(
                self.__class__.__name__))

    def meta_fit(self, X_train, y_train, metastep: 'MetaStep'):
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

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> \
            ('BaseStep', DataContainer):
        """
        Override this to add side effects or change the execution flow before (or after) calling * :func:`~neuraxle.base._FittableStep.fit_transform`.
        The default behavior is to rehash current ids with the step hyperparameters.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)
        """
        if not self.is_initialized:
            self.setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_fit_transform(data_container, context)

        new_self, data_container = self._fit_transform_data_container(data_container, context)

        data_container = self._did_fit_transform(data_container, context)
        data_container = self._did_process(data_container, context)

        return new_self, data_container

    def _will_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> (
            DataContainer, ExecutionContext):
        """
        Apply side effects before fit_transform is called.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        self.invalidate()
        return data_container, context.push(self)

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> \
            ('BaseStep', DataContainer):
        """
        Fit transform data container.

        :param data_container: data container
        :param context: execution context
        :return: (fitted self, data container)
        """
        new_self, out = self.fit_transform(data_container.data_inputs, data_container.expected_outputs)
        data_container.set_data_inputs(out)

        return new_self, data_container

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        """
        Fit, and transform step with the given data inputs, and expected outputs.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on
        :return: (fitted self, tranformed data inputs)
        """
        self.invalidate()

        new_self = self.fit(data_inputs, expected_outputs)
        out = new_self(data_inputs)

        return new_self, out

    def _did_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Apply side effects after fit transform.

        :param data_container: data container
        :param context: execution context
        :return: (fitted self, data container)
        """
        return data_container


class _CustomHandlerMethods:
    """
    A class to represent a step that needs to add special behavior on top of the normal handler methods.
    It allows the step to apply side effects before calling the real handler method.

    Apply additional behavior (mini-batching, parallel processing, etc.) before calling the internal handler methods :
        - :func:`~neuraxle.base._FittableStep._fit_data_container`
        - :func:`~neuraxle.base._FittableStep._fit_transform_data_container`
        - :func:`~neuraxle.base._TransformerStep._transform_data_container`

    .. seealso::
        :class:`~neuraxle.base._FittableStep`,
        :class:`~neuraxle.base._TransformerStep`,
        :class:`~neuraxle.pipeline.MiniBatchSequentialPipeline`,
        :class:`~neuraxle.distributed.streaming.BaseQueuedPipeline`
    """

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseStep':
        """
        Handle fit with a custom handler method for fitting the data container.
        The custom method to override is fit_data_container.
        The custom method fit_data_container replaces _fit_data_container.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)

        .. seealso::
            :class:`~neuraxle.base._FittableStep̀,
            :class:`~neuraxle.data_container.DataContainer`,
            :class:`~neuraxle.base.ExecutionContext`
        """
        if not self.is_initialized:
            self.setup(context)
        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_fit(data_container, context)

        new_self = self.fit_data_container(data_container, context)

        self._did_fit(data_container, context)
        self._did_process(data_container, context)

        return new_self

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> \
            ('BaseStep', DataContainer):
        """
        Handle fit_transform with a custom handler method for fitting, and transforming the data container.
        The custom method to override is fit_transform_data_container.
        The custom method fit_transform_data_container replaces :func:`~neuraxle.base._FittableStep._fit_transform_data_container`.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)

        .. seealso::
            :class:`~neuraxle.base._FittableStep̀,
            :class:`~neuraxle.data_container.DataContainer`,
            :class:`~neuraxle.base.ExecutionContext`
        """
        if not self.is_initialized:
            self.setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_fit_transform(data_container, context)

        new_self, data_container = self.fit_transform_data_container(data_container, context)

        data_container = self._did_fit_transform(data_container, context)
        data_container = self._did_process(data_container, context)

        return new_self, data_container

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Handle transform with a custom handler method for transforming the data container.
        The custom method to override is transform_data_container.
        The custom method transform_data_container replaces :func:`~neuraxle.base._TransformerStep._transform_data_container`.

        :param data_container: the data container to transform
        :param context: execution context
        :return: transformed data container

        .. seealso::
            :class:`~neuraxle.base._TransformerStep̀,
            :class:`~neuraxle.data_container.DataContainer`,
            :class:`~neuraxle.base.ExecutionContext`
        """
        if not self.is_initialized:
            self.setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_transform_data_container(data_container, context)

        data_container = self.transform_data_container(data_container, context)

        data_container = self._did_transform(data_container, context)
        data_container = self._did_process(data_container, context)

        return data_container

    @abstractmethod
    def fit_data_container(self, data_container: DataContainer, context: ExecutionContext):
        """
        Custom fit data container method.

        :param data_container: data container to fit on
        :param context: execution context
        :return: fitted self
        """
        raise NotImplementedError()

    @abstractmethod
    def fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        """
        Custom fit transform data container method.

        :param data_container: data container to fit on
        :param context: execution context
        :return: fitted self, transformed data container
        """
        raise NotImplementedError()

    @abstractmethod
    def transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        """
        Custom transform data container method.

        :param data_container: data container to transform
        :param context: execution context
        :return: transformed data container
        """
        raise NotImplementedError()


class _HasHyperparamsSpace(ABC):
    """
    An internal class to represent a step that has hyperparameter spaces of type :class:`~neuraxle.hyperparams.space.HyperparameterSpace`.
    See :class:`BaseStep`, for a complete step that can be transformed, and fitted inside a :class:`neuraxle.pipeline.Pipeline`.

    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self, hyperparams_space: HyperparameterSpace = None):
        if hyperparams_space is None:
            if hasattr(self, "HYPERPARAMS_SPACE"):
                hyperparams_space = self.HYPERPARAMS_SPACE
            else:
                hyperparams_space = dict()

        self.hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space)

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseTransformer':
        """
        Set step hyperparameters space.

        Example :

        .. code-block:: python

            step.set_hyperparams_space(HyperparameterSpace({
                'hp': RandInt(0, 10)
            }))


        :param hyperparams_space: hyperparameters space
        :return: self

        .. note::
            This is a recursive method that will call :func:`BaseStep._set_hyperparams_space` in the end.
        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
            :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
            :class:`~neuraxle.hyperparams.distributions.HyperparameterDistribution`
            :class:̀_HasChildrenMixin`,
            :func:`BaseStep.apply`,
            :func:`_HasChildrenMixin._apply`,
            :func:`_HasChildrenMixin._get_params`
        """
        self.apply(method='_set_hyperparams_space', hyperparams_space=HyperparameterSpace(hyperparams_space))
        return self

    def _set_hyperparams_space(self, hyperparams_space: Union[Dict, HyperparameterSpace]) -> HyperparameterSpace:
        self._invalidate()
        self.hyperparams_space = HyperparameterSpace(hyperparams_space)
        return self.hyperparams_space

    def update_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseTransformer':
        """
        Update the step hyperparameter spaces without removing the already-set hyperparameters.
        This can be useful to add more hyperparameter spaces to the existing ones without flushing the ones that were already set.

        Example :

        .. code-block:: python

            step.set_hyperparams_space(HyperparameterSpace({
                'learning_rate': LogNormal(0.5, 0.5)
                'weight_decay': LogNormal(0.001, 0.0005)
            }))

            step.update_hyperparams_space(HyperparameterSpace({
                'learning_rate': LogNormal(0.5, 0.1)
            }))

            assert step.get_hyperparams_space()['learning_rate'] == LogNormal(0.5, 0.1)
            assert step.get_hyperparams_space()['weight_decay'] == LogNormal(0.001, 0.0005)


        :param hyperparams_space: hyperparameters space
        :return: self

        .. note::
            This is a recursive method that will call :func:`BaseStep._update_hyperparams_space` in the end.
        .. seealso::
            :func:`~BaseStep.update_hyperparams`,
            :class:`~neuraxle.hyperparams.space.HyperparameterSpace`
        """
        self.apply(method='_update_hyperparams_space',
                   hyperparams_space=HyperparameterSpace(hyperparams_space))
        return self

    def _update_hyperparams_space(self, hyperparams_space: Union[Dict, HyperparameterSpace]) -> HyperparameterSpace:
        self._invalidate()
        hyperparams_space = HyperparameterSpace(hyperparams_space)
        self.hyperparams_space.update(hyperparams_space)
        return self.hyperparams_space

    def get_hyperparams_space(self) -> HyperparameterSpace:
        """
        Get step hyperparameters space.

        Example :

        .. code-block:: python

            step.get_hyperparams_space()

        :return: step hyperparams space

        .. note::
            This is a recursive method that will call :func:`BaseStep._get_hyperparams_space` in the end.
        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
            :class:`~neuraxle.hyperparams.distributions.HyperparameterDistribution`
        """
        results: HyperparameterSpace = self.apply(method='_get_hyperparams_space')
        return results

    def _get_hyperparams_space(self) -> HyperparameterSpace:
        return HyperparameterSpace(self.hyperparams_space)


class _HasHyperparams(ABC):
    """
    An internal class to represent a step that has hyperparameters of type :class:`~neuraxle.hyperparams.space.HyperparameterSamples`.
    See :class:`BaseStep`, for a complete step that can be transformed, and fitted inside a :class:`neuraxle.pipeline.Pipeline`.

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


    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self, hyperparams: HyperparameterSamples = None):
        if hyperparams is None:
            if hasattr(self, "HYPERPARAMS"):
                hyperparams = self.HYPERPARAMS
            else:
                hyperparams = dict()

        self.hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams)

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> 'BaseTransformer':
        """
        Set the step hyperparameters.

        Example :

        .. code-block:: python

            step.set_hyperparams(HyperparameterSamples({
                'learning_rate': 0.10
            }))


        :param hyperparams: hyperparameters
        :return: self

        .. note::
            This is a recursive method that will call :func:`~neuraxle.base._HasHyperparams._set_hyperparams`.
        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
            :class:̀_HasChildrenMixin`,
            :func:`BaseStep.apply`,
            :func:`_HasChildrenMixin._apply`,
            :func:`_HasChildrenMixin._set_train`
        """
        self.apply(method='_set_hyperparams', hyperparams=HyperparameterSamples(hyperparams))
        return self

    def _set_hyperparams(self, hyperparams: Union[HyperparameterSamples, Dict]) -> HyperparameterSamples:
        self._invalidate()
        hyperparams = HyperparameterSamples(hyperparams)
        self.hyperparams = hyperparams if len(hyperparams) > 0 else self.hyperparams
        return self.hyperparams

    def update_hyperparams(self, hyperparams: HyperparameterSamples) -> 'BaseTransformer':
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
        :return: self

        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
            :class:̀_HasChildrenMixin`,
            :func:`BaseStep.apply`,
            :func:`_HasChildrenMixin._apply`,
            :func:`_HasChildrenMixin._update_hyperparams`
        """
        self.apply(method='_update_hyperparams', hyperparams=HyperparameterSamples(hyperparams))
        return self

    def _update_hyperparams(self, hyperparams: Union[Dict, HyperparameterSamples]) -> HyperparameterSamples:
        self.hyperparams.update(HyperparameterSamples(hyperparams))
        return self.hyperparams

    def get_hyperparams(self) -> HyperparameterSamples:
        """
        Get step hyperparameters as :class:`~neuraxle.hyperparams.space.HyperparameterSamples`.

        :return: step hyperparameters

        .. note::
            This is a recursive method that will call :func:`~neuraxle.base._HasHyperparams._get_hyperparams`.
        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
            :class:̀_HasChildrenMixin`,
            :func:`BaseStep.apply`,
            :func:`_HasChildrenMixin._apply`,
            :func:`_HasChildrenMixin._get_hyperparams`
        """
        results: HyperparameterSamples = self.apply(method='_get_hyperparams')
        return results

    def _get_hyperparams(self) -> HyperparameterSamples:
        return HyperparameterSamples(self.hyperparams)

    def set_params(self, **params) -> 'BaseTransformer':
        """
        Set step hyperparameters with a dictionary.

        Example :

        .. code-block:: python

            s.set_params(learning_rate=0.1)
            hyperparams = s.get_params()
            assert hyperparams == {"learning_rate": 0.1}

        :param arbitrary number of arguments for hyperparameters

        .. note::
            This is a recursive method that will call :func:`~neuraxle.base._HasHyperparams._set_params` in the end.
        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
            :class:̀_HasChildrenMixin`,
            :func:`~neuraxle.base._HasRecursiveMethods.apply`,
            :func:`~neuraxle.base._HasChildrenMixin._apply`,
            :func:`~neuraxle.base._HasChildrenMixin._set_params`
        """
        self.apply(method='_set_params', params=HyperparameterSamples(params))
        return self

    def _set_params(self, params: dict) -> HyperparameterSamples:
        self.set_hyperparams(HyperparameterSamples(params))
        return self.hyperparams

    def get_params(self) -> dict:
        """
        Get step hyperparameters as a flat primitive dict.

        Example :

        .. code-block:: python

            s.set_params(learning_rate=0.1)
            hyperparams = s.get_params()
            assert hyperparams == {"learning_rate": 0.1}

        :return: hyperparameters

        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
            :class:̀_HasChildrenMixin`,
            :func:`~neuraxle.base._HasRecursiveMethods.apply`,
            :func:`~neuraxle.base._HasChildrenMixin.apply`,
            :func:`~neuraxle.base._HasHyperparams._get_params`
        """
        results: HyperparameterSamples = self.apply(method='_get_params')
        return results

    def _get_params(self) -> HyperparameterSamples:
        return self.get_hyperparams()


class _HasSavers(ABC):
    """
    An internal class to represent a step that can be saved.
    A step with savers is saved using its list of savers.
    Each saver saves some parts of the step.

    A pipeline can save the step that need to be saved (see :func:`~neuraxle.base._HasSavers.save`) can be saved :

    .. code-block:: python

        step = Pipeline([
            Identity()
        ])
        step.save()
        step = step.load()


    Or, it can also save a full dump that can be reloaded without any source code :

    .. code-block:: python

        step = Identity().set_name('step_name')
        step.save(full_dump=True)
        step = ExecutionContext().load('step_name')


    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self, savers: List[BaseSaver] = None):
        if savers is None:
            savers = []
        self.savers: List[BaseSaver] = savers

    def get_savers(self) -> List[BaseSaver]:
        """
        Get the step savers of a pipeline step.

        :return: step savers

        .. seealso::
            :class:`BaseSaver`
        """
        return self.savers

    def set_savers(self, savers: List[BaseSaver]) -> 'BaseTransformer':
        """
        Set the step savers of a pipeline step.

        :return: self

        .. seealso::
            :class:`BaseSaver`
        """
        self.savers: List[BaseSaver] = savers
        return self

    def add_saver(self, saver: BaseSaver) -> 'BaseTransformer':
        """
        Add a step saver of a pipeline step.

        :return: self

        .. seealso::
            :class:`BaseSaver`
        """
        self.savers.append(saver)
        return self

    def should_save(self) -> bool:
        """
        Returns true if the step should be saved.
        If the step has been initialized and invalidated, then it must be saved.

        A step is invalidated when any of the following things happen :

            * a mutation has been performed on the step :func:`~.mutate`
            * an hyperparameter has changed func:`~neuraxle.base._HasHyperparams.set_hyperparams`
            * an hyperparameter space has changed func:`~neuraxle.base._HasHyperparamsSpace.set_hyperparams_space`
            * a call to the fit method func:`~neuraxle.base._FittableStep.handle_fit`
            * a call to the fit_transform method func:`~neuraxle.base._FittableStep.handle_fit_transform`
            * the step name has changed func:`~neuraxle.base.BaseStep.set_name`

        :return: if the step should be saved
        """
        return self.is_invalidated and self.is_initialized

    def save(self, context: ExecutionContext, full_dump=False) -> 'BaseTransformer':
        """
        Save step using the execution context to create the directory to save the step into.
        The saving happens by looping through all of the step savers in the reversed order.

        Some savers just save parts of objects, some save it all or what remains.
        The :class:`ExecutionContext`.stripped_saver has to be called last because it needs a
        stripped version of the step.

        :param context: context to save from
        :param full_dump: save full pipeline dump to be able to load everything without source code (false by default).
        :return: self

        .. seealso::
            :class:`ExecutionContext`,
            :class:`BaseSaver`
        """
        context = context.push(self)
        self.is_invalidated = False

        def _initialize_if_needed(step):
            if not step.is_initialized:
                step.setup(context=context)
            if not step.is_initialized:
                raise NotImplementedError(f"The `setup` method of the following class "
                                          f"failed to set `self.is_initialized` to True: {step.__class__.__name__}.")
            return RecursiveDict()

        def _invalidate(step):
            step._invalidate()
            return RecursiveDict()

        if full_dump:
            # initialize and invalidate steps to make sure that all steps will be saved
            self.apply(method=_initialize_if_needed)
            self.apply(method=_invalidate)

        context.mkdir()
        stripped_step = copy(self)

        # A final "visitor" saver will save anything that
        # wasn't saved customly after stripping the rest.
        savers_with_provided_default_stripped_saver = [context.stripped_saver] + self.savers

        for saver in reversed(savers_with_provided_default_stripped_saver):
            # Each saver strips the step a bit more if needs be.
            stripped_step = saver.save_step(stripped_step, context)

        return stripped_step

    def load(self, context: ExecutionContext, full_dump=False) -> 'BaseTransformer':
        """
        Load step using the execution context to create the directory of the saved step.
        Warning:

        :param context: execution context to load step from
        :param full_dump: save full dump bool
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
        has_been_loaded_by_a_saver: bool = False
        for saver in savers:
            # Each saver unstrips the step a bit more if needed
            if saver.can_load(loaded_self, context):
                loaded_self = saver.load_step(loaded_self, context)
                has_been_loaded_by_a_saver = True
            else:
                has_been_loaded_by_a_saver and warnings.warn(
                    'Cannot Load Step {0} ({1}:{2}) With Step Saver {3}.'.format(
                        context.get_path(),
                        self.name,
                        self.__class__.__name__,
                        saver.__class__.__name__)
                )
                break
        return loaded_self


class _HasHashers(ABC):
    """
    An internal class to represent a step that has hashers.
    Most step rehash after every transformations to update the summary id, and the current ids inside the :class:`DataContainer`.
    Hashers hash by hyperparameters, and source code.

    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self, hashers: List[BaseHasher] = None):
        if hashers is None:
            hashers = [HashlibMd5Hasher()]

        self.hashers: List[BaseHasher] = hashers

    def set_hashers(self, hashers: List[BaseHasher]) -> '_HasHashers':
        self.hashers: List[BaseHasher] = hashers
        return self

    def summary_hash(self, data_container: DataContainer) -> str:
        """
        Hash data inputs, current ids, and hyperparameters together using self.hashers.
        This is used to create unique ids for the data checkpoints.

        :param data_container: data container
        :return: hashed current ids

        .. seealso::
            :class:`~neuraxle.checkpoints.Checkpoint`
        """
        if data_container.summary_id is None:
            data_container.set_summary_id(data_container.hash_summary())

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
        :return: hashed current ids

        .. seealso::
            :class:`~neuraxle.checkpoints.Checkpoint`
        """
        current_ids = data_container.current_ids
        for h in self.hashers:
            current_ids = h.hash(current_ids, self.hyperparams, data_container.data_inputs)

        return current_ids

    def hash_data_container(self, data_container):
        """
        Hash data container using self.hashers.

        #. Hash current ids with hyperparams.
        #. Hash summary id with hyperparams.

        :param data_container: the data container to transform
        :return: transformed data container
        """
        current_ids = self.hash(data_container)
        data_container.set_current_ids(current_ids)

        summary_id = self.summary_hash(data_container)
        data_container.set_summary_id(summary_id)

        return data_container


class _HasMutations(ABC):
    """
    An internal class to represent a step that can be mutated.
    A step can replace some of its method by others.
    For example, you might want to reverse a step, and replace the transform method by the inverse transform method.

    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :func:`~neuraxle.base._TransformerStep.transform`,
        :func:`~neuraxle.base._TransformerStep.inverse_transform`
    """

    def __init__(self):
        self.pending_mutate: ('BaseTransformer', str, str) = (None, None, None)

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
        self._invalidate()
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
            self._invalidate()

        except AttributeError as e:
            if warn:
                import warnings
                warnings.warn(repr(e))

        return new_base_step

    def will_mutate_to(self, new_base_step: 'BaseTransformer' = None, new_method: str = None,
                       method_to_assign_to: str = None) -> 'BaseTransformer':
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
        :param method_to_assign_to: if it is not None, upon calling ``mutate``, the method_to_affect will be the one that is used on the provided new_base_step.
        :param new_method: if it is not None, upon calling ``mutate``, the new_method will be the one that is used on the provided new_base_step.
        :return: self
        """
        self._invalidate()

        if new_method is None or method_to_assign_to is None:
            new_method = method_to_assign_to = "transform"  # No changes will be applied (transform will stay transform).

        self.pending_mutate = (new_base_step, new_method, method_to_assign_to)

        return self


class _CouldHaveContext:
    """
    Step that can have a context.
    It has "has service assertions" to ensure that the context has registered all the necessary services.

    A context can be injected with the with_context method:

    .. code-block:: python

        context = ExecutionContext(root=tmpdir)
        service = SomeService()
        context.set_service_locator({SomeBaseService: service})

        p = Pipeline([
            SomeStep().assert_has_services(SomeBaseService)
        ]).with_context(context=context)

    Or alternatively,

    .. code-block:: python

        p = Pipeline([
            RegisterSomeService(),
            SomeStep().assert_has_services_at_execution(SomeBaseService)
        ])

    Context services can be used inside any step with handler methods:

    .. code-block:: python

        class SomeStep(ForceHandleMixin, Identity):
            def __init__(self):
                Identity.__init__(self)
                ForceHandleMixin.__init__(self)

            def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
                service: SomeBaseService = context.get_service(SomeBaseService)
                service.service_method(data_container.data_inputs)
                return data_container


    .. seealso::
        :class:`~neuraxle.base.BaseTransformer`,
        :class:`~neuraxle.base._TransformerStep`,
    """

    def with_context(self, context: ExecutionContext):
        """
        An higher order step to inject a context inside a step.
        A step with a context forces the pipeline to use that context through handler methods.
        This is useful for dependency injection because you can register services inside the :class:`ExecutionContext`.
        It also ensures that the context has registered all the necessary services.

        .. code-block:: python

            context = ExecutionContext(root=tmpdir)
            context.set_service_locator(ServiceLocator().services) # where services is of type Dict[Type, object]

            p = WithContext(Pipeline([
                SomeStep().with_assertion_has_services(SomeBaseService)
            ]), context)


        .. seealso::
            :class:`BaseStep`,
            :class:`ExecutionContext`,
            :class:`BaseTransformer`
        """
        return StepWithContext(wrapped=self, context=context)

    def assert_has_services(self, *service_assertions: List[Type]) -> 'GlobalyRetrievableServiceAssertionWrapper':
        """
        Set all service assertions to be made at the root of the pipeline and before processing the step.

        :param service_assertions: base types that need to be available in the execution context at the root of the pipeline
        :type service_assertions: List[Type]
        """
        return GlobalyRetrievableServiceAssertionWrapper(wrapped=self, service_assertions=service_assertions)

    def assert_has_services_at_execution(self, *service_assertions: List[Type]) -> 'LocalServiceAssertionWrapper':
        """
        Set all service assertions to be made before processing the step.

        :param service_assertions: base types that need to be available in the execution context before the execution of the step
        :type service_assertions: List[Type]
        """
        return LocalServiceAssertionWrapper(wrapped=self, service_assertions=service_assertions)


class BaseTransformer(
    _CouldHaveContext,
    _HasMutations,
    _HasHyperparamsSpace,
    _HasHyperparams,
    _HasHashers,
    _HasSavers,
    _HasRecursiveMethods,
    _TransformerStep,
    ABC
):
    """
    Base class for a pipeline step that can only be transformed.

    Every step can be saved using its savers of type :class:`BaseSaver` (see :class:`~neuraxle.base._HasHashers` for more info).
    Most step hash data inputs with hyperparams after every transformations to update the current ids inside the :class:`~neuraxle.data_container.DataContainer` (see :class:`~neuraxle.base._HasHashers` for more info).
    Every step has hyperparemeters, and hyperparameters spaces that can be set before the learning process begins (see :class:`_HasHyperparams`, and :class:`_HasHyperparamsSpace` for more info).

    Example usage :

    .. code-block:: python

        class AddN(BaseTransformer):
            def __init__(self, add=1):
                super().__init__(hyperparams=HyperparameterSamples({ 'add': add }))

            def transform(self, data_inputs):
                if not isinstance(data_inputs, np.ndarray):
                    data_inputs = np.array(data_inputs)

                return data_inputs + self.hyperparams['add']

            def inverse_transform(self, data_inputs):
                if not isinstance(data_inputs, np.ndarray):
                    data_inputs = np.array(data_inputs)

                return data_inputs - self.hyperparams['add']


    .. note:: All heavy initialization logic should be done inside the *setup* method (e.g.: things inside GPU), and NOT in the constructor.
    .. seealso::
        :class:`~neuraxle.pipeline.Pipeline`,
        :class:`~neuraxle.base.BaseTransformer`,
        :class:`~neuraxle.base._TransformerStep`,
        :class:`~neuraxle.base._HasHyperparamsSpace`,
        :class:`~neuraxle.base._HasHyperparams`,
        :class:`~neuraxle.base._HasHashers`,
        :class:`~neuraxle.base._HasSavers`,
        :class:`~neuraxle.base._HasMutations`,
        :class:`~neuraxle.base._HasRecursiveMethods`,
        :class:`~neuraxle.base._HasDependencies`,
        :class:`~neuraxle.base.NonFittableMixin`,
        :class:`~neuraxle.base.NonTransformableMixin`
    """

    def __init__(
            self,
            hyperparams: HyperparameterSamples = None,
            hyperparams_space: HyperparameterSpace = None,
            name: str = None,
            savers: List[BaseSaver] = None,
            hashers: List[BaseHasher] = None
    ):
        _TransformerStep.__init__(self)
        _HasRecursiveMethods.__init__(self)
        _HasHyperparams.__init__(self, hyperparams=hyperparams)
        _HasHyperparamsSpace.__init__(self, hyperparams_space=hyperparams_space)
        _HasSavers.__init__(self, savers=savers)
        _HasHashers.__init__(self, hashers=hashers)
        _HasMutations.__init__(self)
        _CouldHaveContext.__init__(self)

        if name is None:
            name = self.__class__.__name__
        self.name: str = name
        self._invalidate()

        self.is_initialized = False
        self.is_train: bool = True

    def setup(self, context: ExecutionContext = None) -> 'BaseTransformer':
        """
        Initialize the step before it runs. Only from here and not before that heavy things should be created
        (e.g.: things inside GPU), and NOT in the constructor.

        .. warning::
            The setup method is called once for each step when handle_fit, handle_fit_transform or handle_transform is called.
            The setup method is executed only if is self.is_initialized is False
            A setup function should set the self.is_initialized to True when called.

        :param context: execution context
        :return: self
        """
        self.is_initialized = True
        return self

    def invalidate(self) -> 'BaseTransformer':
        """
        Invalidate a step, and all of its children. Invalidating a step makes it eligible to be saved again.

        A step is invalidated when any of the following things happen :
            * a mutation has been performed on the step: func:`~neuraxle.base._HasMutations.mutate`
            * an hyperparameter has changed func: `~neuraxle.base._HasHyperparams.set_hyperparams`
            * an hyperparameter space has changed func: `~neuraxle.base._HasHyperparamsSpace.set_hyperparams_space`
            * a call to the fit method func:`~neuraxle.base._FittableStep.handle_fit`
            * a call to the fit_transform method func:`~neuraxle.base._FittableStep.handle_fit_transform`
            * the step name has changed func:`~neuraxle.base.BaseStep.set_name`

        :return: self

        .. note::
            This is a recursive method used in :class:̀_HasChildrenMixin`.

        .. seealso::
            :func:`~neuraxle.base._HasRecursiveMethods.apply`,
            :func:`~neuraxle.base._HasChildrenMixin._apply`
        """
        self.apply(method='_invalidate')
        return self

    def _invalidate(self):
        self.is_invalidated = True
        return RecursiveDict()

    def _teardown(self) -> 'BaseTransformer':
        """
        Teardown step after program execution. Inverse of setup, and it should clear memory.
        Override this method if you need to clear memory.

        :return: self
        """
        self.is_initialized = False
        return self

    def __del__(self):
        try:
            self._teardown()
        except Exception:
            print(traceback.format_exc())

    def set_train(self, is_train: bool = True):
        """
        This method overrides the method of BaseStep to also consider the wrapped step as well as self.
        Set pipeline step mode to train or test.

        .. note::
            This is a recursive method used in :class:̀_HasChildrenMixin`.
        .. seealso::
            :func:`~neuraxle.base._HasRecursiveMethods.apply`,
            :func:`~neuraxle.base._HasChildrenMixin._apply`
            :func:`~neuraxle.base.BaseTransformer._set_train`
        """
        self.apply(method='_set_train', is_train=is_train)
        return self

    def _set_train(self, is_train) -> RecursiveDict:
        self.is_train = is_train
        return RecursiveDict()

    def set_name(self, name: str):
        """
        Set the name of the pipeline step.

        :param name: a string.
        :return: self

        .. note::
            A step name is the same value as the one in the keys of :py:attr:`~neuraxle.pipeline.Pipeline.steps_as_tuple`
        """
        self.name = name
        self.invalidate()
        return self

    def get_name(self) -> str:
        """
        Get the name of the pipeline step.

        :return: the name, a string.

        .. note:: A step name is the same value as the one in the keys of :class:`Pipeline`.steps_as_tuple
        """
        return self.name

    def get_step_by_name(self, name):
        if self.name == name :
            return self
        return None

    def reverse(self) -> 'BaseTransformer':
        """
        The object will mutate itself such that the ``.transform`` method (and of all its underlying objects
        if applicable) be replaced by the ``.inverse_transform`` method.

        Note: the reverse may fail if there is a pending mutate that was set earlier with ``.will_mutate_to``.

        :return: a copy of self, reversed. Each contained object will also have been reversed if self is a pipeline.

        .. seealso::
            :func:`~neuraxle.base._TransformerStep.inverse_transform`
        """
        return self.mutate(new_method="inverse_transform", method_to_assign_to="transform", warn=False)

    def __reversed__(self) -> 'BaseTransformer':
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


def _sklearn_to_neuraxle_step(step) -> BaseTransformer:
    if hasattr(step, '_get_param_names') and hasattr(step, '_more_tags') \
            and hasattr(step, '_check_n_features') and hasattr(step, '_validate_data'):
        import neuraxle.steps.sklearn
        step = neuraxle.steps.sklearn.SKLearnWrapper(step)
        step.set_name(step.get_wrapped_sklearn_predictor().__class__.__name__)
    return step


class BaseStep(_FittableStep, BaseTransformer, ABC):
    """
    Base class for a transformer step that can also be fitted.

    If a step is not fittable, you can inherit from :class:`BaseTransformer` instead.
    If a step is not transformable, you can inherit from :class:`NonTransformableMixin`.
    A step should only change its state inside :func:`~neuraxle.base._FittableStep.fit` or :func:`~neuraxle.base._FittableStep.fit_transform` (see :class:`_FittableStep` for more info).
    Every step can be saved using its savers of type :class:`BaseSaver` (see :class:`~neuraxle.base._HasHashers` for more info).
    Most step hash data inputs with hyperparams after every transformations to update the current ids inside the :class:`DataContainer` (see :class:`~neuraxle.base._HasHashers` for more info).
    Every step has hyperparemeters, and hyperparameters spaces that can be set before the learning process begins (see :class:`_HasHyperparams`, and :class:`_HasHyperparamsSpace` for more info).

    Example usage :

    .. code-block:: python

        class Normalize(BaseStep):
            def __init__(self):
                BaseStep.__init__(self)
                self.mean = None
                self.std = None

            def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
                self._calculate_mean_std(data_inputs)
                return self

            def _calculate_mean_std(self, data_inputs):
                self.mean = np.array(data_inputs).mean(axis=0)
                self.std = np.array(data_inputs).std(axis=0)

            def fit_transform(self, data_inputs, expected_outputs=None):
                self.fit(data_inputs, expected_outputs)
                return self, (np.array(data_inputs) - self.mean) / self.std

            def transform(self, data_inputs):
                if self.mean is None or self.std is None:
                    self._calculate_mean_std(data_inputs)
                return (np.array(data_inputs) - self.mean) / self.std

        p = Pipeline([
            Normalize()
        ])

        p, outputs = p.fit_transform(data_inputs, expected_outputs)


    .. note:: All heavy initialization logic should be done inside the *setup* method (e.g.: things inside GPU), and NOT in the constructor.
    .. seealso::
        :class:`~neuraxle.pipeline.Pipeline`,
        :class:`~neuraxle.base._TransformerStep`,
        :class:`~neuraxle.base._HasHyperparamsSpace`,
        :class:`~neuraxle.base._HasHyperparams`,
        :class:`~neuraxle.base._HasHashers`,
        :class:`~neuraxle.base._HasSavers`,
        :class:`~neuraxle.base._HasMutations`,
        :class:`~neuraxle.base._HasRecursiveMethods`,
        :class:`~neuraxle.base.NonFittableMixin`,
        :class:`~neuraxle.base.NonTransformableMixin`
    """
    pass


class MixinForBaseTransformer:
    """
    Any steps/transformers within a pipeline that inherits of this class should implement BaseStep/BaseTransformer and initialize it before any mixin. This class checks that its the case at initialization.
    """

    def __init__(self):
        self._ensure_basetransformer_init_called()

    def _ensure_basetransformer_init_called(self):
        """
        Assert that BaseTransformer's init method has been called.
        """
        assert isinstance(self, BaseTransformer)
        if not all(map(lambda x: hasattr(self, x), ('name', 'savers', 'is_initialized', 'is_train', 'is_invalidated'))):
            raise RuntimeError('Please initialize Mixins in the good order. This Mixin should be initialized after '
                               'BaseTransformer.')


class _HasChildrenMixin(MixinForBaseTransformer):
    """
    Mixin to add behavior to the steps that have children (sub steps).

    .. seealso::
        :class:`MixinForBaseTransformer`
        :class:`~neuraxle.base.MetaStepMixin`,
        :class:`~neuraxle.base.TruncableSteps`
    """

    def apply(self, method: Union[str, Callable], ra: _RecursiveArguments = None, *args, **kwargs) -> RecursiveDict:
        """
        Apply method to root, and children steps.
        Split the root, and children values inside the arguments of type RecursiveDict.

        :param method: str or callable function to apply
        :param ra: recursive arguments
        :return:
        """
        ra: _RecursiveArguments = _RecursiveArguments(ra=ra, *args, **kwargs)
        results: RecursiveDict = self._apply_self(method=method, ra=ra)
        results: RecursiveDict = self._apply_childrens(results=results, method=method, ra=ra)

        return results

    def _apply_self(self, method: Union[str, Callable], ra: _RecursiveArguments):
        terminal_ra: _RecursiveArguments = ra[None]
        self_results: RecursiveDict = BaseStep.apply(self, method=method, ra=terminal_ra)
        return self_results

    def _apply_childrens(self, results: RecursiveDict, method: Union[str, Callable],
                         ra: _RecursiveArguments) -> RecursiveDict:
        for children in self.get_children():
            children_results: RecursiveDict = children.apply(method=method, ra=ra[children.get_name()])
            results[children.get_name()] = children_results

        return results

    @abstractmethod
    def get_children(self) -> List[BaseStep]:
        """
        Get the list of all the childs for that step.

        :return:
        """
        pass


class MetaStepMixin(_HasChildrenMixin):
    """
    A class to represent a step that wraps another step. It can be used for many things.
    For example, :class:`~neuraxle.steps.loop.ForEachDataInput` adds a loop before any calls to the wrapped step :

    .. code-block:: python

        class ForEachDataInput(MetaStepMixin, BaseStep):
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
        :class:`~neuraxle.steps.loop.ForEachDataInput`,
        :class:`~neuraxle.metaopt.random.BaseCrossValidationWrapper`,
        :class:`~neuraxle.steps.caching.ValueCachingWrapper`,
        :class:`~neuraxle.steps.loop.StepClonerForEachDataInput`
    """

    def __init__(self, wrapped: BaseTransformer = None, savers: List[BaseSaver] = None):
        if savers is None:
            savers = []
        MixinForBaseTransformer.__init__(self)
        self.wrapped: BaseTransformer = _sklearn_to_neuraxle_step(wrapped)
        savers.append(MetaStepJoblibStepSaver())
        self.savers.extend(savers)

    def set_step(self, step: BaseTransformer) -> BaseStep:
        """
        Set wrapped step to the given step.

        :param step: new wrapped step
        :return: self
        """
        self._invalidate()
        self.wrapped: BaseTransformer = _sklearn_to_neuraxle_step(step)
        return self

    def get_step(self) -> BaseStep:
        """
        Get wrapped step

        :return: self.wrapped
        """
        return self.wrapped

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        previous_summary_id = data_container.summary_id
        new_self, data_container = super().handle_fit_transform(data_container, context)
        data_container.set_summary_id(previous_summary_id)
        data_container.set_summary_id(self.summary_hash(data_container))

        return new_self, data_container

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext):
        previous_summary_id = data_container.summary_id
        data_container = super().handle_transform(data_container, context)
        data_container.set_summary_id(previous_summary_id)
        data_container.set_summary_id(self.summary_hash(data_container))

        return data_container

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
        return self, data_container

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext):
        self.wrapped = self.wrapped.handle_fit(data_container, context)
        return self

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        data_container = self.wrapped.handle_transform(data_container, context)
        return data_container

    def _inverse_transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        data_container = self.wrapped.handle_inverse_transform(data_container, context)
        return data_container

    def fit_transform(self, data_inputs, expected_outputs=None):
        self.wrapped, data_inputs = self.wrapped.fit_transform(data_inputs, expected_outputs)
        return self, data_inputs

    def fit(self, data_inputs, expected_outputs=None):
        self.wrapped = self.wrapped.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        data_inputs = self.wrapped.transform(data_inputs)
        return data_inputs

    def inverse_transform(self, data_inputs):
        data_inputs = self.wrapped.inverse_transform(data_inputs)
        return data_inputs

    def should_resume(self, data_container: DataContainer, context: ExecutionContext):
        context = context.push(self)
        if isinstance(self.wrapped, ResumableStepMixin) and self.wrapped.should_resume(data_container, context):
            return True
        return False

    def resume(self, data_container: DataContainer, context: ExecutionContext):
        context = context.push(self)
        if not isinstance(self.wrapped, ResumableStepMixin):
            raise Exception('cannot resume steps that don\' inherit from ResumableStepMixin')

        data_container = self.wrapped.resume(data_container, context)
        data_container = self._did_process(data_container, context)
        return data_container

    def get_children(self) -> List[BaseStep]:
        """
        Get the list of all the childs for that step.
        :class:`_HasChildrenMixin` calls this method to apply methods to all of the childs for that step.

        :return: list of child steps

        .. seealso::
            :class:`_HasChildrenMixin`
        """
        return [self.wrapped]

    def get_step_by_name(self, name):
        if self.name == name:
            return self
        if self.wrapped.name == name:
            return self.wrapped
        return self.wrapped.get_step_by_name(name)

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=False) -> 'BaseTransformer':
        """
        Mutate self, and self.wrapped. Please refer to :func:`~neuraxle.base._HasMutations.mutate` for more information.

        :param new_method: the method to replace transform with, if there is no pending ``will_mutate_to`` call.
        :param method_to_assign_to: the method to which the new method will be assigned to, if there is no pending ``will_mutate_to`` call.
        :param warn: (verbose) wheter or not to warn about the inexistence of the method.
        :return: self, a copy of self, or even perhaps a new or different BaseStep object.
        """
        new_self = super().mutate(new_method, method_to_assign_to, warn)
        new_self.wrapped = self.wrapped.mutate(new_method, method_to_assign_to, warn)

        return new_self

    def will_mutate_to(self, new_base_step: 'BaseTransformer' = None, new_method: str = None,
                       method_to_assign_to: str = None) -> 'BaseTransformer':
        """
        Add pending mutate self, self.wrapped. Please refer to :func:`~neuraxle.base._HasMutations.will_mutate_to` for more information.

        :param new_base_step: if it is not None, upon calling ``mutate``, the object it will mutate to will be this provided new_base_step.
        :param method_to_assign_to: if it is not None, upon calling ``mutate``, the method_to_affect will be the one that is used on the provided new_base_step.
        :param new_method: if it is not None, upon calling ``mutate``, the new_method will be the one that is used on the provided new_base_step.
        :return: self
        """
        new_self = super().will_mutate_to(new_base_step, new_method, method_to_assign_to)
        return new_self

    def __repr__(self):
        output = self.__class__.__name__ + "(\n\twrapped=" + repr(
            self.wrapped) + "," + "\n\thyperparameters=" + pprint.pformat(
            self.hyperparams) + "\n)"
        return output


class MetaStep(MetaStepMixin, BaseStep):
    def __init__(
            self,
            wrapped: BaseTransformer = None,
            hyperparams: HyperparameterSamples = None,
            hyperparams_space: HyperparameterSpace = None,
            name: str = None,
            savers: List[BaseSaver] = None,
            hashers: List[BaseHasher] = None
    ):
        BaseStep.__init__(
            self,
            hyperparams=hyperparams,
            hyperparams_space=hyperparams_space,
            name=name,
            savers=savers,
            hashers=hashers
        )
        MetaStepMixin.__init__(self, wrapped=wrapped)


class MetaStepJoblibStepSaver(JoblibStepSaver):
    """
    Custom saver for meta step mixin.
    """

    def __init__(self):
        JoblibStepSaver.__init__(self)

    def save_step(self, step: 'MetaStep', context: ExecutionContext) -> MetaStep:
        """
        Save MetaStepMixin.

        #. Save wrapped step.
        #. Strip wrapped step form the meta step mixin.
        #. Save meta step with wrapped step savers.

        :param step: meta step to save
        :param context: execution context
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

    def load_step(self, step: 'MetaStep', context: ExecutionContext) -> 'MetaStep':
        """
        Load MetaStepMixin.

        #. Loop through all of the sub steps savers, and only load the sub steps that have been saved.
        #. Refresh steps

        :param step: step to load
        :param context: execution context
        :return: loaded truncable steps
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


NamedTupleList = List[Union[Tuple[str, BaseTransformer], BaseTransformer]]


class NonFittableMixin(MixinForBaseTransformer):
    """
    A pipeline step that requires no fitting: fitting just returns self when called to do no action.
    Note: fit methods are not implemented
    """

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext):
        return self

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        return self, self._transform_data_container(data_container, context)

    def fit(self, data_inputs, expected_outputs=None) -> 'NonFittableMixin':
        """
        Don't fit.

        :param data_inputs: the data that would normally be fitted on.
        :param expected_outputs: the data that would normally be fitted on.
        :return: self
        """
        return self


class NonTransformableMixin(MixinForBaseTransformer):
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

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        return self._fit_data_container(data_container, context), data_container

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Do nothing - return the same data.

        :param data_container: data container
        :param context: execution context
        :return: data container
        """
        return data_container

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
        :param context: execution context
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
        if hasattr(step, 'steps'):
            del step.steps
            del step.steps_as_tuple

        return step

    def load_step(self, step: 'TruncableSteps', context: ExecutionContext) -> 'TruncableSteps':
        """
        #. Loop through all of the sub steps savers, and only load the sub steps that have been saved.
        #. Refresh steps

        :param step: step to load
        :param context: execution context
        :return: loaded truncable steps
        """
        step.steps_as_tuple = []

        for step_name, savers in step.sub_steps_savers:
            if (savers is None):
                # keep step as it is if it hasn't been saved
                step.steps_as_tuple.append((step_name, step[step_name]))
            else:
                # Load each sub step with their savers
                sub_step_to_load = Identity(name=step_name, savers=savers)
                sub_step = sub_step_to_load.load(context)
                step.steps_as_tuple.append((step_name, sub_step))

        operation = getattr(step, '_refresh_steps', None)
        if callable(operation):
            step._refresh_steps()

        return step


class TruncableSteps(_HasChildrenMixin, BaseStep, ABC):
    """
    Step that contains multiple steps. :class:`Pipeline` inherits form this class.
    It is possible to truncate this step * :func:`~neuraxle.base.TruncableSteps.__getitem__`

    * self.steps contains the actual steps
    * self.steps_as_tuple contains a list of tuple of step name, and step

    .. seealso::
        :class:`~neuraxle.pipeline.Pipeline`,
        :class:`~neuraxle.union.FeatureUnion`
    """

    def __init__(
            self,
            steps_as_tuple: NamedTupleList,
            hyperparams: HyperparameterSamples = dict(),
            hyperparams_space: HyperparameterSpace = dict(),
            mute_step_renaming_warning: bool = True,
    ):
        BaseStep.__init__(self, hyperparams=hyperparams, hyperparams_space=hyperparams_space)
        _HasChildrenMixin.__init__(self)
        self.warn_step_renaming = not mute_step_renaming_warning
        self.set_steps(steps_as_tuple)
        self.set_savers([TruncableJoblibStepSaver()] + self.savers)

    def are_steps_before_index_the_same(self, other: 'TruncableSteps', index: int) -> bool:
        """
        Returns true if self.steps before index are the same as other.steps before index.

        :param other: other truncable steps to compare
        :param index: max step index to compare

        :return: bool
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
        :param index: step index
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
        :return:
        """
        steps_as_tuple = self._wrap_non_base_steps(steps_as_tuple)
        self.steps_as_tuple: NamedTupleList = self._patch_missing_names(steps_as_tuple)
        self._refresh_steps()

    def get_children(self) -> List[BaseStep]:
        """
        Get the list of sub step inside the step with children.

        :return: children steps
        """
        return list(self.values())

    def get_step_by_name(self, name):
        for step in self.values():
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

            step = _sklearn_to_neuraxle_step(step)

            if class_name is None:
                class_name = step.get_name()

            wrapped.append((class_name, step))
        return wrapped

    def _patch_missing_names(self, steps_as_tuple: NamedTupleList) -> NamedTupleList:
        """
        Make sure that each sub step has a unique name, and add a name to the sub steps that don't have one already.

        :param steps_as_tuple: a NamedTupleList
        :return: a NamedTupleList with fixed names
        """
        # TODO: document more the type of the `steps as tuple`.
        names_yet = set()
        patched = []
        for class_name, step in steps_as_tuple:
            _name = class_name
            if class_name in names_yet:
                if self.warn_step_renaming:
                    warnings.warn(
                        "Named pipeline tuples must be unique. "
                        "Will rename '{}' because it already exists.".format(class_name))

                _name = self._rename_step(step_name=_name, class_name=class_name, names_yet=names_yet)
                step.set_name(_name)

            step = (_name, step)
            names_yet.add(step[0])
            patched.append(step)
        self._invalidate()
        return patched

    def _rename_step(self, step_name, class_name, names_yet: set):
        """
        Rename step by adding a number suffix after the class name.
        Ensure uniqueness with the names yet parameter.

        :param step_name: step name
        :param class_name: class name
        :param names_yet: names already taken
        :return: new step name
        """
        # Add suffix number to name if it is already used to ensure name uniqueness.
        i = 1
        while step_name in names_yet:
            step_name = class_name + str(i)
            i += 1
        self._invalidate()
        return step_name

    def _refresh_steps(self):
        """
        Private method to refresh inner state after having edited ``self.steps_as_tuple``
        (recreate ``self.steps`` from ``self.steps_as_tuple``).
        """
        self._invalidate()
        self.steps: OrderedDict = OrderedDict(self.steps_as_tuple)
        for name, step in self.items():
            step.name = name

    def should_save(self):
        """
        Returns if the step needs to be saved or not.

        :return: If self or any of his sub steps should be saved, returns True.

        .. seealso::
            :class:`TruncableJoblibStepSaver`
        """
        if super().should_save():
            return True

        for _, step in self.items():
            if step.should_save():
                return True
        return False

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=False) -> 'BaseTransformer':
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
            new_base_step = BaseStep.mutate(self, new_method, method_to_assign_to, warn)
            self.pending_mutate = (new_base_step, self.pending_mutate[1], self.pending_mutate[2])

            new_base_step.steps_as_tuple = [
                (
                    k,
                    v.mutate(new_method, method_to_assign_to, warn)
                )
                for k, v in new_base_step.steps_as_tuple
            ]
            new_base_step._refresh_steps()
            return new_base_step
        else:
            # Since we're remplacing ourselves with a new step, we don't have to call mutate on our childrens since
            # they won't exist afterward.
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
        :param new_step: step
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
        :return: truncated self
        .. seealso::
            :class:`~neuraxle.data_container.DataContainer`,
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
        :return: new truncable steps with concatenated steps
        """
        self.set_steps(self.steps_as_tuple + other.steps_as_tuple)
        return self

    def items(self) -> ItemsView:
        """
        Returns all of the steps as tuples items (step_name, step).

        :return: step items tuple : (step name, step)
        """
        return self.steps.items()

    def keys(self) -> KeysView:
        """
        Returns the step names.

        :return: list of step names
        """
        return self.steps.keys()

    def values(self) -> ValuesView:
        """
        Get step values.

        :return: all of the steps
        """
        return self.steps.values()

    def append(self, item: Tuple[str, 'BaseTransformer']) -> 'TruncableSteps':
        """
        Add an item to steps as tuple.

        :param item: item tuple (step name, step)
        :return: self
        """
        self.steps_as_tuple.append(item)
        self._refresh_steps()
        return self

    def pop(self) -> 'BaseTransformer':
        """
        Pop the last step.

        :return: last step
        """
        return self.popitem()[-1]

    def popitem(self, key=None) -> Tuple[str, 'BaseTransformer']:
        """
        Pop the last step, or the step with the given key

        :param key: step name to pop, or None
        :return: last step item

        """
        if key is None:
            item = self.steps_as_tuple.pop()
            self._refresh_steps()
        else:
            item = key, self.steps.pop(key)
            self.steps_as_tuple = list(self.steps.items())
        return item

    def popfront(self) -> 'BaseTransformer':
        """
        Pop the first step.

        :return: first step
        """
        return self.popfrontitem()[-1]

    def popfrontitem(self) -> Tuple[str, 'BaseTransformer']:
        """
        Pop the first step.

        :return: first step item
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

        :return: if truncable steps ends with the given step type
        """
        return isinstance(self[-1], step_type)

    def __repr__(self):

        output = self.__class__.__name__ + '\n' \
                 + "(\n\t" + super(TruncableSteps, self).__repr__() \
                 + "(\n\t\t" + pprint.pformat(self.steps_as_tuple) \
                 + "\t\n)" \
                 + "\n)"

        return output

    def __str__(self):
        return self.__repr__()


class ResumableStepMixin(MixinForBaseTransformer):
    """
    Mixin to add resumable function to a step, or a class that can be resumed, for example a checkpoint on disk.
    """

    @abstractmethod
    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        """
        Returns True if a step can be resumed with the given the data container and execution context.
        See Checkpoint class documentation for more details on how a resumable checkpoint works.

        :param data_container: data container to resume from
        :param context: execution context to resume from
        :return: if we can resume
        """
        raise NotImplementedError()

    @abstractmethod
    def resume(self, data_container: DataContainer, context: ExecutionContext):
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
        BaseStep.__init__(self, name=name, savers=savers)
        NonFittableMixin.__init__(self)
        NonTransformableMixin.__init__(self)


class TransformHandlerOnlyMixin(MixinForBaseTransformer):
    """
    A pipeline step that only requires the implementation of _transform_data_container.

    .. seealso::
        :class:`BaseStep`,
        :class:`MixinForBaseTransformer`
        :class:`NonFittableMixin`
    """

    @abstractmethod
    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Transform data container with the given execution context.

        :param data_container: data container
        :param context: execution context
        :return: transformed data container
        """
        raise NotImplementedError('Must implement _transform_data_container in {0}'.format(self.name))

    def transform(self, data_inputs) -> 'HandleOnlyMixin':
        raise Exception(
            'Transform method is not supported for {0}, because it inherits from HandlerMixin. Please use handle_transform instead.'.format(
                self.name))


class HandleOnlyMixin(MixinForBaseTransformer):
    """
    A pipeline step that only requires the implementation of handler methods :
        - _transform_data_container
        - _fit_transform_data_container
        - _fit_data_container

    If forbids only implementing fit or transform or fit_transform without the handles. So it forces the handles.

    .. seealso::
        :class:`BaseStep`,
        :class:`MixinForBaseTransformer`
        :class:`TransformHandlerOnlyMixin`,
        :class:`NonTransformableMixin`,
        :class:`NonFittableMixin`,
        :class:`ForceHandleMixin`,
        :class:`ForceHandleOnlyMixin`
    """

    @abstractmethod
    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseTransformer':
        raise NotImplementedError('Must implement _fit_data_container in {0}'.format(self.name))

    @abstractmethod
    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        raise NotImplementedError('Must implement _transform_data_container in {0}'.format(self.name))

    @abstractmethod
    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> \
            ('BaseTransformer', DataContainer):
        raise NotImplementedError('Must implement _fit_transform_data_container in {0}'.format(self.name))

    def transform(self, data_inputs) -> 'HandleOnlyMixin':
        raise Exception(
            'Transform method is not supported for {0}, because it inherits from HandleOnlyMixin. Please use handle_transform instead.'.format(
                self.name))

    def fit(self, data_inputs, expected_outputs=None) -> 'HandleOnlyMixin':
        raise Exception(
            'Fit method is not supported for {0}, because it inherits from HandleOnlyMixin. Please use handle_fit instead.'.format(
                self.name))

    def fit_transform(self, data_inputs, expected_outputs=None) -> 'HandleOnlyMixin':
        raise Exception(
            'Fit transform method is not supported for {0}, because it inherits from HandleOnlyMixin. Please use handle_fit_transform instead.'.format(
                self.name))


class ForceHandleMixin(MixinForBaseTransformer):
    """
    A step that automatically calls handle methods in the transform, fit, and fit_transform methods.
    A class which inherits from ForceHandleMixin can't use BaseStep's  _fit_data_container, _fit_transform_data_container and _transform_data_container. They must be redefined; failure to do so will trigger an Exception on initialisation (and would create infinite loop if these checks were not there).

    .. seealso::
        :class:`BaseStep`,
        :class:'MixinForBaseTransformer'
        :class:`HandleOnlyMixin`,
        :class:`TransformHandlerOnlyMixin`,
        :class:`NonTransformableMixin`,
        :class:`NonFittableMixin`,
        :class:`ForceHandleOnlyMixin`
    """

    def __init__(self, cache_folder=None):
        MixinForBaseTransformer.__init__(self)
        if cache_folder is None:
            cache_folder = DEFAULT_CACHE_FOLDER
        self.cache_folder = cache_folder
        if isinstance(self, _FittableStep):
            self._ensure_method_overriden("_fit_data_container", _FittableStep)
            self._ensure_method_overriden("_fit_transform_data_container", _FittableStep)
        self._ensure_method_overriden("_transform_data_container", _TransformerStep)

    def _ensure_method_overriden(self, method_name, original_cls):
        """
        Asserts that a given method of current instance overrides the default one defined in a given class. We assume that current instance inherits from the given class but do not test it (MixinForBaseTransformer test inheritance to BaseTransfromer already).

        :param method_name:
        :param original_cls:
        :return:
        """
        if original_cls.__dict__[method_name] == getattr(self, method_name).__func__:
            raise NotImplementedError(
                f"The ForceHandleMixin class overrides fit, transform and fit_transform to force a call on their handler method counterparts. Failure to redefine basic _fit_data_container, _transform_data_container and _fit_transform_data_container methods can cause an infinite loop. Please define {method_name} in {self.__class__.__name__}.")

    def transform(self, data_inputs) -> Iterable:
        """
        Using :func:`~neuraxle.base._TransformerStep.handle_transform`, transform data inputs.

        :param data_inputs: data inputs
        :return: outputs
        """
        context, data_container = self._encapsulate_data(
            data_inputs, expected_outputs=None, execution_mode=ExecutionMode.TRANSFORM)

        data_container = self.handle_transform(data_container, context)

        return data_container.data_inputs

    def fit(self, data_inputs, expected_outputs=None) -> 'HandleOnlyMixin':
        """
        Using :func:`~neuraxle.base._FittableStep.handle_fit`, fit step with the given data inputs, and expected outputs.

        :param data_inputs: data inputs
        :return: fitted self
        """
        context, data_container = self._encapsulate_data(data_inputs, expected_outputs, ExecutionMode.FIT)
        new_self = self.handle_fit(data_container, context)

        return new_self

    def fit_transform(self, data_inputs, expected_outputs=None) -> \
            Tuple['HandleOnlyMixin', Iterable]:
        """
        Using :func:`~neuraxle.base._FittableStep.handle_fit_transform`, fit and transform step with the given data inputs, and expected outputs.

        :param data_inputs: data inputs
        :return: fitted self, outputs
        """
        context, data_container = self._encapsulate_data(data_inputs, expected_outputs, ExecutionMode.FIT_TRANSFORM)
        new_self, data_container = self.handle_fit_transform(data_container, context)

        return new_self, data_container.data_inputs

    def _encapsulate_data(self, data_inputs, expected_outputs=None, execution_mode=None) -> \
            Tuple[ExecutionContext, DataContainer]:
        """
        Encapsulate data with :class:`~neuraxle.data_container.DataContainer`.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs
        :param execution_mode: execution mode
        :return: execution context, data container
        """
        data_container = DataContainer(data_inputs=data_inputs, expected_outputs=expected_outputs)
        context = ExecutionContext(root=self.cache_folder, execution_mode=execution_mode)

        return context, data_container


class ForceHandleIdentity(ForceHandleMixin, Identity):
    """
    An identity step which forces usage of handler methods. Mostly used in unit tests.
    Useful when you only want to define few methods such as _will_process or _did_process.
    """
    def __init__(self):
        Identity.__init__(self)
        ForceHandleMixin.__init__(self)


class ForceHandleOnlyMixin(ForceHandleMixin, HandleOnlyMixin):
    """
    A step that automatically calls handle methods in the transform, fit, and fit_transform methods.

    It also requires the implementation of handler methods :
        - _transform_data_container
        - _fit_transform_data_container
        - _fit_data_container

    .. seealso::
        :class:`BaseStep`,
        :class:`HandleOnlyMixin`,
        :class:`TransformHandlerOnlyMixin`,
        :class:`NonTransformableMixin`,
        :class:`NonFittableMixin`,
        :class:`ForceHandleMixin`
    """

    def __init__(self, cache_folder=None):
        HandleOnlyMixin.__init__(self)
        ForceHandleMixin.__init__(self, cache_folder)


class IdentityHandlerMethodsMixin(ForceHandleOnlyMixin):
    """
    A step that has a default implementation for all handler methods.

    It is useful for steps that only the following methods :
        - :func:`~neuraxle.base._FittableStep.will_fit`
        - :func:`~neuraxle.base._TransformerStep.will_transform`
        - :func:`~neuraxle.base._FittableStep.will_fit_transform`
        - :func:`~neuraxle.base._TransformerStep.will_process`
        - :func:`~neuraxle.base._TransformerStep.did_process`
        - :func:`~neuraxle.base._FittableStep.did_fit`
        - :func:`~neuraxle.base._TransformerStep.did_transform`
        - :func:`~neuraxle.base._FittableStep.did_fit_transform`

    .. seealso::
        :class:`BaseStep`,
        :class:`HandleOnlyMixin`,
        :class:`TransformHandlerOnlyMixin`,
        :class:`NonTransformableMixin`,
        :class:`NonFittableMixin`,
        :class:`ForceHandleMixin`,
        :class:`HandleOnlyMixin`,
        :class:`ForceHandleOnlyMixin`
    """

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> \
            ('BaseTransformer', DataContainer):
        return self

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        return data_container

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> \
            ('BaseTransformer', DataContainer):
        return self, data_container


class EvaluableStepMixin(MixinForBaseTransformer):
    """
    A step that can be evaluated with the scoring functions.

    .. seealso::
        :class:`BaseStep`
        :class:'MixinForBaseTransformer'
    """

    @abstractmethod
    def get_score(self):
        raise NotImplementedError()


class FullDumpLoader(Identity):
    """
    Identity step that can load the full dump of a pipeline step.
    Used by :func:`~neuraxle.base._HasSavers.load`.

    Usage example:

    .. code-block:: python

        saved_step = FullDumpLoader(
            name=path,
            stripped_saver=self.stripped_saver
        ).load(context_for_loading, True)


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
        :return: loaded step
        """
        if not context.stripped_saver.can_load(self, context):
            raise Exception('Cannot Load Full Dump For Step {}'.format(os.path.join(context.get_path(), self.name)))

        loaded_self = context.stripped_saver.load_step(self, context)

        context.pop()
        return loaded_self.load(context, full_dump)


class AssertionMixin(ForceHandleMixin):
    def __init__(self):
        ForceHandleMixin.__init__(self)

    @abstractmethod
    def _assert(self, data_container: DataContainer, context: ExecutionContext):
        pass


class WillProcessAssertionMixin(AssertionMixin):
    def _will_process(self, data_container: DataContainer, context: ExecutionContext) -> (
            DataContainer, ExecutionContext):
        """
        Calls self._assert(data_container,context)
        """
        data_container, context = super()._will_process(data_container, context)
        self._assert(data_container, context)

        return data_container, context


class DidProcessAssertionMixin(AssertionMixin):
    def _did_process(self, data_container: DataContainer, context: ExecutionContext) -> (
            DataContainer, ExecutionContext):
        """
        Calls self._assert(data_container,context)
        """
        data_container, context = super()._did_process(data_container, context)
        self._assert(data_container, context)

        return data_container, context


class AssertExpectedOutputIsNoneMixin(WillProcessAssertionMixin):
    def _assert(self, data_container: DataContainer, context: ExecutionContext):
        eo_empty = (data_container.expected_outputs is None) or all(v is None for v in data_container.expected_outputs)
        if not eo_empty:
            raise AssertionError(
                f"Expected datacontainer.expected_outputs to be a `None` or a list of `None`. Received {data_container.expected_outputs}.")

class AssertExpectedOutputIsNotNoneMixin(WillProcessAssertionMixin):
    def _assert(self, data_container: DataContainer, context: ExecutionContext):
        eo_empty = (data_container.expected_outputs is None) or all(v is None for v in data_container.expected_outputs)
        if eo_empty:
            raise AssertionError(
                f"Expected datacontainer.expected_outputs to not be a `None` or a list of `None`. Received {data_container.expected_outputs}.")

class AssertExpectedOutputIsNoneStep(AssertExpectedOutputIsNoneMixin, Identity):
    def __init__(self):
        Identity.__init__(self)
        AssertExpectedOutputIsNoneMixin.__init__(self)


class AssertExpectedOutputIsNone(AssertExpectedOutputIsNoneMixin, Identity):
    def __init__(self):
        Identity.__init__(self)
        AssertExpectedOutputIsNoneMixin.__init__(self)


class LocalServiceAssertionWrapper(WillProcessAssertionMixin, MetaStep):
    """
    Is used to assert the presence of service at execution time for a given step
    """

    def __init__(self, wrapped: BaseTransformer = None, service_assertions: List[Type] = None,
                 savers: List[BaseSaver] = None):
        MetaStep.__init__(self, wrapped=wrapped, savers=savers)
        WillProcessAssertionMixin.__init__(self)

        if service_assertions is None:
            service_assertions = []
        self.service_assertions = service_assertions

    def _assert(self, data_container: DataContainer, context: ExecutionContext):
        """
        Assert self.local_service_assertions are present in the context.
        """
        self.do_assert_has_services(context)

    def do_assert_has_services(self, context: ExecutionContext):
        """
        Assert that all the necessary services are provided in the execution context.

        :param context: The ExecutionContext for which we test the presence of service.
        """
        for has_service_assertion in self.service_assertions:
            if not context.has_service(service_abstract_class_type=has_service_assertion):
                exception_message: str = '{} dependency missing in the ExecutionContext. Please register the service {} inside the ExecutionContext.\n'.format(
                    has_service_assertion.__name__,
                    has_service_assertion.__name__
                )
                step_method_message: str = 'You can do so by calling register_service, or set_services on any step.\n'
                execution_context_methods_messsage: str = 'There is also the option to register all services inside the ExecutionContext'
                raise AssertionError(exception_message + step_method_message + execution_context_methods_messsage)


class GlobalyRetrievableServiceAssertionWrapper(LocalServiceAssertionWrapper):
    """
    Is used to assert the presence of service at the start of the pipeline AND at execution time for a given step.
    """

    def _global_assert_has_services(self, context: ExecutionContext) -> RecursiveDict:
        """
        Intended to be used in a .apply('_global_assert_has_services') call from the outside.
        Is used to test the presence of services at the root of the pipeline.

        See also GlobalServiceAssertionExecutorMixin._apply_service_assertions
        :params context : the execution context
        """
        self.do_assert_has_services(context)
        return RecursiveDict()


class GlobalServiceAssertionExecutorMixin(WillProcessAssertionMixin):
    """
    Any step which inherit of this class will test globaly retrievable service assertion of itself and all its children on a will_process call.
    """

    def _assert(self, data_container: DataContainer, context: ExecutionContext):
        """
        Calls _global_assert_has_services on GlobalyRetrievableServiceAssertionWrapper instances that are (recursively) children of this node.
        :param context: The ExecutionContext for which we test the presence of service.
        """
        self.apply('_global_assert_has_services', context=context)


class _WithContextStepSaver(BaseSaver):
    """
    Custom saver for steps that have an :class:`ExecutionContext`.
    Loading will inject the saved dependencies inside the :class`ExecutionContext`.

    .. seealso::
        :class:`_HasContext`,
        :class:`BaseSaver`,
        :class:`ExecutionContext`
    """

    def load_step(self, step: 'StepWithContext', context: ExecutionContext) -> 'StepWithContext':
        """
        Load a step with a context by setting the context as the loading context.

        :param step: step with context
        :param context: execution context to load from
        :return: loaded step with context
        """
        step.context = context
        warnings.warn("Warning! the loading of a StepWithContext instance overrides the context attribute with the one provided at loading.")
        return step

    def save_step(self, step: 'StepWithContext', context: ExecutionContext) -> 'StepWithContext':
        """
        If needed, remove parents of a step with context before saving.

        :param step: step with context
        :param context: execution context to load from
        :return: saved step with context
        """
        del step.context
        return step

    def can_load(self, step: 'StepWithContext', context: 'ExecutionContext'):
        return True


class StepWithContext(GlobalServiceAssertionExecutorMixin, MetaStep):
    def __init__(self, wrapped: 'BaseTransformer', context: ExecutionContext, raise_if_not_root: bool = True):
        MetaStep.__init__(self, wrapped=wrapped, savers=[_WithContextStepSaver()])
        GlobalServiceAssertionExecutorMixin.__init__(self)
        self.context = context
        self.raise_if_not_root = raise_if_not_root

    def _will_process(self, data_container: DataContainer, context: ExecutionContext) -> (
            DataContainer, ExecutionContext):
        """
        Inject the given context and test service assertions (if any are appliable) before processing the wrapped step.

        :param data_container: data container to process
        :return: data container, execution context
        """
        if self.raise_if_not_root and len(context) > 0:
            raise AssertionError('StepWithContext should be at the root of the pipeline.')

        data_container, context = GlobalServiceAssertionExecutorMixin._will_process(self, data_container, self.context)

        return data_container, context
