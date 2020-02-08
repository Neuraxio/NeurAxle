"""
Neuraxle's Checkpoint Classes
====================================
The checkpoint classes used by the checkpoint pipeline runner

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

import os
import pickle
from abc import abstractmethod, ABC
from enum import Enum
from typing import List, Tuple, Any

from neuraxle.base import ResumableStepMixin, BaseStep, ExecutionContext, \
    ExecutionMode, NonTransformableMixin, NonFittableMixin, Identity
from neuraxle.data_container import DataContainer, ListDataContainer


class DataCheckpointType(Enum):
    DATA_INPUT = 'di'
    EXPECTED_OUTPUT = 'eo'


class BaseCheckpointer(NonFittableMixin, NonTransformableMixin, BaseStep):
    """
    Base class to implement a step checkpoint or data container checkpoint.

    :class:`Checkpoint` uses many BaseCheckpointer to checkpoint both data container checkpoints, and step checkpoints.

    BaseCheckpointer has an execution mode so there could be different checkpoints for each execution mode (fit, fit_transform or transform).

    .. seealso::
        * :class:`Checkpoint`
    """

    def __init__(
            self,
            execution_mode: ExecutionMode
    ):
        BaseStep.__init__(self)
        self.execution_mode = execution_mode

    def is_for_execution_mode(self, execution_mode: ExecutionMode) -> bool:
        """
        Returns true if the checkpointer should be used with the given execution mode.

        :param execution_mode: execution mode (fit, fit_transform, or transform)
        :type execution_mode: ExecutionMode
        :return: if the checkpointer should be used
        :rtype: bool
        """
        if execution_mode == ExecutionMode.FIT:
            return self.execution_mode in [
                ExecutionMode.FIT,
                ExecutionMode.FIT_OR_FIT_TRANSFORM,
                ExecutionMode.FIT_OR_FIT_TRANSFORM_OR_TRANSFORM
            ]

        if execution_mode == ExecutionMode.FIT_TRANSFORM:
            return self.execution_mode in [
                ExecutionMode.FIT_TRANSFORM,
                ExecutionMode.FIT_OR_FIT_TRANSFORM,
                ExecutionMode.FIT_OR_FIT_TRANSFORM_OR_TRANSFORM
            ]

        if execution_mode == ExecutionMode.TRANSFORM:
            return self.execution_mode in [
                ExecutionMode.TRANSFORM,
                ExecutionMode.FIT_OR_FIT_TRANSFORM_OR_TRANSFORM
            ]

        return execution_mode == ExecutionMode.FIT_OR_FIT_TRANSFORM_OR_TRANSFORM

    @abstractmethod
    def save_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Save the data container or fitted step checkpoint with the given data container, and context.
        Returns the data container checkpoint, or latest data container.

        :param data_container: data container to save data container or fitted steps
        :type data_container: neuraxle.data_container.DataContainer
        :param context: context to save data container or fitted steps
        :type context: ExecutionContext
        :return: saved data container
        :rtype: neuraxle.data_container.DataContainer
        """
        raise NotImplementedError()

    @abstractmethod
    def read_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Read the data container checkpoint with self.data_checkpointer.
        Returns a new data container loaded with all the data inputs,
        and expected outputs for each current id in the given data container.

        :param data_container: data container containing the current_ids to read checkpoint for
        :type data_container: neuraxle.data_container.DataContainer
        :param context: context to read checkpoint for
        :type context: ExecutionContext
        :return: the data container checkpoint
        :rtype: neuraxle.data_container.DataContainer
        """
        raise NotImplementedError()

    @abstractmethod
    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        raise NotImplementedError()


class StepSavingCheckpointer(BaseCheckpointer):
    """
    StepCheckpointer is used by the Checkpoint step to save the fitted steps contained in the context of type ExecutionContext.

    By default, StepCheckpointer saves the fitted steps when the execution mode is either FIT, or FIT_TRANSFORM :

    .. code:: python

        StepCheckpointer(ExecutionMode.FIT_OR_FIT_TRANSFORM)
        # is equivalent to :
        StepCheckpointer()

    """

    def __init__(
            self,
            execution_mode: ExecutionMode = ExecutionMode.FIT_OR_FIT_TRANSFORM,
    ):
        BaseCheckpointer.__init__(self, execution_mode=execution_mode)

    def read_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        # violating ISP for guillaume
        return data_container

    def save_checkpoint(
            self,
            data_container: DataContainer,
            context: ExecutionContext
    ) -> DataContainer:
        if self.is_for_execution_mode(context.get_execution_mode()):
            # TODO: save the context by execution mode AND data container ids / summary
            context.copy().save()

        return data_container

    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        # TODO: change this when we support multiple execution modes and data container ids / summary
        return True


class Checkpoint(NonFittableMixin, NonTransformableMixin, ResumableStepMixin, BaseStep):
    """
    Resumable Checkpoint Step to load, and save both data checkpoints, and step checkpoints.
    Checkpoint uses a list of step checkpointers(List[StepCheckpointer]), and data checkpointers(List[BaseCheckpointer]).

    Data Checkpoints save the state of the data container (transformed data inputs, and expected outputs)
    for the current execution mode (fit, fit_transform, or transform).

    Step Checkpoints save the state of the fitted steps before the checkpoint
    for the current execution mode (fit or fit_transform).

    By default(no arguments specified), the Checkpoint step saves the step checkpoints for any fit or fit transform,
    and saves a different data checkpoint with pickle data container checkpointers :

    .. code:: python

        Checkpoint(
            all_checkpointers=[
                StepSavingCheckpointer(),
                MiniDataCheckpointerWrapper(
                    data_input_checkpointer=PickleMiniDataCheckpointer(),
                    expected_output_checkpointer=PickleMiniDataCheckpointer()
                )
            ]
        )

    .. seealso::
        * :class:`BaseStep`
        * :func:`ResumablePipeline._load_checkpoint`
        * :class:`ResumableStepMixin`
        * :class:`NonFittableMixin`
        * :class:`NonTransformableMixin`
    """

    def __init__(
            self,
            all_checkpointers: List[BaseCheckpointer] = None,
    ):
        BaseStep.__init__(self)
        self.all_checkpointers = all_checkpointers

    def _fit_data_container(self, data_container, context) -> 'Checkpoint':
        """
        Saves step, and data checkpointers for the FIT execution mode.

        :param data_container: data container for creating the data checkpoint
        :param context: context for creating the step checkpoint
        :return: saved data container
        :rtype: neuraxle.data_container.DataContainer
        """
        self.save_checkpoint(data_container, context)
        return self

    def _transform_data_container(self, data_container, context):
        """
        Saves step, and data checkpointers for the TRANSORM execution mode.

        :param data_container: data container for creating the data checkpoint
        :param context: context for creating the step checkpoint
        :return: saved data container
        :rtype: neuraxle.data_container.DataContainer
        """
        return self.save_checkpoint(data_container, context)

    def _fit_transform_data_container(self, data_container, context) -> Tuple['Checkpoint', DataContainer]:
        """
        Saves step, and data checkpointers for the FIT_TRANSORM execution mode.

        :param data_container: data container for creating the data checkpoint
        :param context: context for creating the step checkpoint
        :return: saved data container
        :rtype: neuraxle.data_container.DataContainer
        """
        return self, self.save_checkpoint(data_container, context)

    def save_checkpoint(self, data_container: DataContainer, context: ExecutionContext):
        """
        Saves step, and data checkpointers for the current execution mode.

        :param data_container: data container for creating the data checkpoint
        :param context: context for creating the step checkpoint
        :return: saved data container
        :rtype: neuraxle.data_container.DataContainer
        """
        for checkpointer in self.all_checkpointers:
            checkpointer.save_checkpoint(data_container, context)

        return data_container

    def read_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Read data checkpoint for the current execution mode using self.data_checkpointers.

        :param data_container: data container to load checkpoint from
        :param context: execution mode to load checkpoint from
        :return: loaded data container checkpoint
        :rtype: neuraxle.data_container.DataContainer
        """
        context = context.push(self)
        for checkpointer in self.all_checkpointers:
            if checkpointer.is_for_execution_mode(context.get_execution_mode()):
                data_container = checkpointer.read_checkpoint(data_container, context)

        return data_container

    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        """
        Returns True if all of the execution mode data checkpointers can be resumed.

        :param context: execution context
        :param data_container: data to resume
        :return: if we can resume the checkpoint
        :rtype: bool
        """
        context = context.push(self)
        for checkpointer in self.all_checkpointers:
            if checkpointer.is_for_execution_mode(context.get_execution_mode()):
                if not checkpointer.should_resume(data_container, context):
                    return False

        return True


class BaseSummaryCheckpointer(ABC):
    """
    Summary Checkpointer to create a summary file that contains the list of all of the checkpoint current ids.

    A summary checkpointer must be wrapped with a :class:`MiniDataCheckpointerWrapper` to be added to a :class:`Checkpoint`
    :py:attr:`~Checkpoint.data_checkpointers` :

    .. code:: python

        Checkpoint(
            all_checkpointers=[
                StepSavingCheckpointer(),
                MiniDataCheckpointerWrapper(
                    summary_checkpointer=TextSummaryCheckpointer(),
                    data_input_checkpointer=PickleMiniDataCheckpointer(),
                    expected_output_checkpointer=PickleMiniDataCheckpointer()
                )
            ]
        )

    .. seealso::
        * :class:`BaseMiniDataCheckpointer`
        * :class:`MiniDataCheckpointerWrapper`
        * :class:`PickleMiniDataCheckpointer`
    """

    @abstractmethod
    def save_summary(self, checkpoint_path, data_container: DataContainer):
        """
        Save data checkpoint with the given current_id, and data.

        :param checkpoint_path: checkpoint path for saving
        :type checkpoint_path: str
        :param current_id: current id to checkpoint
        :type current_id: str
        :param data: data to checkpoint
        :type data: Any
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def read_summary(self, checkpoint_path, data_container: DataContainer) -> List[str]:
        """
        Read data checkpoint with the given current_id, and data.

        :param checkpoint_path: checkpoint path to read
        :type checkpoint_path: str
        :param current_id: current id to read checkpoint for
        :type current_id: str
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def checkpoint_exists(self, checkpoint_path, data_container: DataContainer) -> bool:
        """
        Returns if checkpoint exists with the given path, and current id.

        :param checkpoint_path: checkpoint path to read
        :type checkpoint_path: str
        :param current_id: current id to read checkpoint for
        :type current_id: str
        :return:
        """
        raise NotImplementedError()


class BaseMiniDataCheckpointer(ABC):
    """
    Mini Data Checkpoint that uses pickle to create a checkpoint for a current id, and a data input or an expected output.

    A mini data checkpointer must be wrapped with a :class:`MiniDataCheckpointerWrapper` to be added to a :class:`Checkpoint`
    :py:attr:`~Checkpoint.data_checkpointers` :

    .. code:: python

        Checkpoint(
            all_checkpointers=[
                StepSavingCheckpointer(),
                MiniDataCheckpointerWrapper(
                    summary_checkpointer=PickleSummaryCheckpointer(),
                    data_input_checkpointer=PickleMiniDataCheckpointer(),
                    expected_output_checkpointer=PickleMiniDataCheckpointer()
                )
            ]
        )

    .. seealso::
        * :class:`BaseMiniDataCheckpointer`
        * :class:`MiniDataCheckpointerWrapper`
        * :class:`PickleMiniDataCheckpointer`
    """

    @abstractmethod
    def save_checkpoint(self, checkpoint_path, current_id, data):
        """
        Save data checkpoint with the given current_id, and data.

        :param checkpoint_path: checkpoint path for saving
        :type checkpoint_path: str
        :param current_id: current id to checkpoint
        :type current_id: str
        :param data: data to checkpoint
        :type data: Any
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def read_checkpoint(self, checkpoint_path, current_id) -> Tuple:
        """
        Read data checkpoint with the given current_id, and data.

        :param checkpoint_path: checkpoint path to read
        :type checkpoint_path: str
        :param current_id: current id to read checkpoint for
        :type current_id: str
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def checkpoint_exists(self, checkpoint_path, current_id) -> bool:
        """
        Returns if checkpoint exists with the given path, and current id.

        :param checkpoint_path: checkpoint path to read
        :type checkpoint_path: str
        :param current_id: current id to read checkpoint for
        :type current_id: str
        :return:
        """
        raise NotImplementedError()


class NullMiniDataCheckpointer(BaseMiniDataCheckpointer):
    def set_checkpoint_type(self, checkpoint_type: DataCheckpointType):
        pass

    def save_checkpoint(self, checkpoint_path, current_id, data):
        pass

    def read_checkpoint(self, checkpoint_path, current_id) -> Tuple:
        return None

    def checkpoint_exists(self, checkpoint_path, current_id) -> bool:
        return True


class TextFileSummaryCheckpointer(BaseSummaryCheckpointer):
    """
    Summary Checkpointer that uses a txt file to create a summary file that contains the list of all of the checkpoint current ids.
    A summary checkpoint file is a txt file that contains the list of all of the current ids of the checkpoint.

    .. seealso::
        * :class:`BaseSummaryCheckpointer`
        * :class:`BaseMiniDataCheckpointer`
        * :class:`MiniDataCheckpointerWrapper`
        * :class:`PickleMiniDataCheckpointer`
    """

    def save_summary(self, checkpoint_path, data_container: DataContainer):
        """
        Save summary checkpoint file.

        :param checkpoint_path: checkpoint path for saving
        :type checkpoint_path: str
        :param data_container: checkpoint data container
        :type data_container: DataContainer
        :return:
        """
        with open(os.path.join(checkpoint_path, '{0}.txt'.format(data_container.summary_id)), 'w+') as file:
            lines = [str(cuid) + '\n' for cuid in data_container.current_ids]
            lines[-1] = str(data_container.current_ids[-1])
            file.writelines(lines)

    def read_summary(self, checkpoint_path, data_container: DataContainer) -> List[str]:
        """
        Read current ids inside a summary checkpoint file.

        :param checkpoint_path: checkpoint path for saving
        :type checkpoint_path: str
        :param data_container: checkpoint data container
        :type data_container: DataContainer
        :return: checkpoint current ids
        :rtype: List[str]
        """
        with open(os.path.join(checkpoint_path, '{0}.txt'.format(data_container.summary_id)), 'r') as file:
            current_ids = file.readlines()
        return [cuid.strip() for cuid in current_ids]

    def checkpoint_exists(self, checkpoint_path, data_container: DataContainer) -> bool:
        """
        Returns true if the checkpoint summary file exists.

        :param checkpoint_path: checkpoint path for saving
        :type checkpoint_path: str
        :param data_container: checkpoint data container
        :type data_container: DataContainer
        :return:
        """
        return os.path.exists(
            os.path.join(
                checkpoint_path, '{0}.txt'.format(data_container.summary_id)
            )
        )


class PickleMiniDataCheckpointer(BaseMiniDataCheckpointer):
    """
    Mini Data Checkpoint that uses pickle to create a pickle checkpoint file for a current id, and a data input or expected output.

    A mini data checkpointer must be wrapped with a :class:`MiniDataCheckpointerWrapper` to be added to a :class:`Checkpoint`
    :py:attr:`~Checkpoint.data_checkpointers` :

    .. code:: python

        Checkpoint(
            all_checkpointers=[
                StepSavingCheckpointer(),
                MiniDataCheckpointerWrapper(
                    summary_checkpointer=TextSummaryCheckpointer(),
                    data_input_checkpointer=PickleMiniDataCheckpointer(),
                    expected_output_checkpointer=PickleMiniDataCheckpointer()
                )
            ]
        )

    .. seealso::
        * :class:`BaseMiniDataCheckpointer`
        * :class:`MiniDataCheckpointerWrapper`
    """

    def set_checkpoint_type(self, checkpoint_type: DataCheckpointType):
        """
        Set file name suffix for checkpoint.

        :param checkpoint_type: checkpoint file name suffix
        :type checkpoint_type: str
        :return:
        """
        self.file_name_suffix = checkpoint_type.value

    def save_checkpoint(self, checkpoint_path: str, current_id, data):
        """
        Save the given current id, data input, and expected output using pickle.dump.

        :param checkpoint_path: checkpoint path
        :type checkpoint_path: str
        :param current_id: checkpoint current id
        :type current_id: str
        :param data: data to checkpoint
        :type data: Any
        :return:
        """
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        with open(self.get_checkpoint_filename_path_for_current_id(checkpoint_path, current_id), 'wb') as file:
            pickle.dump(data, file)

    def read_checkpoint(self, checkpoint_path: str, current_id) -> Any:
        """
        Read the data inputs, and expected outputs for the given current id using pickle.load.

        :param checkpoint_path: checkpoint folder path
        :type checkpoint_path: str
        :param current_id: checkpoint current id
        :type current_id: str
        :return: tuple(current_id, checkpoint_data_input, checkpoint_expected_output)
        :rtype: Any
        """
        with open(self.get_checkpoint_filename_path_for_current_id(checkpoint_path, current_id), 'rb') as file:
            return pickle.load(file)

    def get_checkpoint_filename_path_for_current_id(self, checkpoint_path: str, current_id: str) -> str:
        """
        Get the checkpoint file path for a data input id.

        :param checkpoint_path: checkpoint folder path
        :type checkpoint_path: str
        :param current_id: checkpoint current id
        :type current_id: str
        :return: path
        :rtype: str
        """
        return os.path.join(checkpoint_path, '{0}.pickle'.format(current_id))

    def checkpoint_exists(self, checkpoint_path: str, current_id: str) -> bool:
        """
        Get the checkpoint file path for a data input id.

        :param checkpoint_path: checkpoint folder path
        :type checkpoint_path: str
        :param current_id: checkpoint current id
        :type current_id: str
        :return: path
        :rtype: str
        """
        return os.path.exists(
            os.path.join(checkpoint_path, '{0}.pickle'.format(current_id))
        )


class MiniDataCheckpointerWrapper(BaseCheckpointer):
    """
    A :class:`BaseCheckpointer` to checkpoint data inputs, and expected outputs with mini data checkpointers.

    .. code:: python

        MiniDataCheckpointerWrapper(
            data_input_checkpointer=PickleMiniDataCheckpointer(),
            expected_output_checkpointer=PickleMiniDataCheckpointer()
        )

    .. seealso::
        * :class:`BaseMiniDataCheckpointer`
        * :class:`BaseCheckpointer`
    """

    def __init__(
            self,
            summary_checkpointer: BaseSummaryCheckpointer,
            data_input_checkpointer: BaseMiniDataCheckpointer,
            expected_output_checkpointer: BaseMiniDataCheckpointer = None
    ):
        execution_mode = ExecutionMode.FIT_OR_FIT_TRANSFORM_OR_TRANSFORM  # TODO: analyse if we need this or not ?
        BaseCheckpointer.__init__(self, execution_mode)

        self.summary_checkpointer = summary_checkpointer
        self.data_input_checkpointer: BaseMiniDataCheckpointer = data_input_checkpointer

        if expected_output_checkpointer is None:
            expected_output_checkpointer = NullMiniDataCheckpointer()

        self.expected_output_checkpointer: BaseMiniDataCheckpointer = expected_output_checkpointer

    def save_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Save data container data inputs with :py:attr:`~data_input_checkpointer`.
        Save data container expected outputs with :py:attr:`~expected_output_checkpointer`.

        :param data_container: data container to checkpoint
        :type data_container: neuraxle.data_container.DataContainer
        :param context: execution context to checkpoint from
        :type context: ExecutionContext
        :return:
        """
        if not self.is_for_execution_mode(context.get_execution_mode()):
            return data_container

        context.mkdir()

        self.summary_checkpointer.save_summary(
            checkpoint_path=context.get_path(),
            data_container=data_container
        )

        for current_id, data_input, expected_output in data_container:
            self.data_input_checkpointer.save_checkpoint(
                checkpoint_path=self._get_data_input_checkpoint_path(context),
                current_id=current_id,
                data=data_input
            )

            self.expected_output_checkpointer.save_checkpoint(
                checkpoint_path=self._get_expected_output_checkpoint_path(context),
                current_id=current_id,
                data=expected_output
            )

        return data_container

    def read_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Read data container data inputs checkpoint with :py:attr:`~data_input_checkpointer`.
        Read data container expected outputs checkpoint with :py:attr:`~expected_output_checkpointer`.

        :param data_container: data container to read checkpoint for
        :type data_container: neuraxle.data_container.DataContainer
        :param context: execution context to read checkpoint from
        :type context: ExecutionContext
        :return: data container checkpoint
        :rtype: neuraxle.data_container.DataContainer
        """
        data_container_checkpoint = ListDataContainer.empty()

        current_ids = self.summary_checkpointer.read_summary(
            checkpoint_path=context.get_path(),
            data_container=data_container
        )

        for current_id in current_ids:
            data_input = self.data_input_checkpointer.read_checkpoint(
                checkpoint_path=self._get_data_input_checkpoint_path(context),
                current_id=current_id
            )

            expected_output = self.expected_output_checkpointer.read_checkpoint(
                checkpoint_path=self._get_expected_output_checkpoint_path(context),
                current_id=current_id
            )

            data_container_checkpoint.append(
                current_id,
                data_input,
                expected_output
            )

        return data_container_checkpoint

    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        """
        Returns if the whole data container has been checkpointed.

        :param data_container: data container to read checkpoint for
        :type data_container: neuraxle.data_container.DataContainer
        :param context: execution context to read checkpoint from
        :type context: ExecutionContext
        :return: data container checkpoint
        :rtype: neuraxle.data_container.DataContainer
        """
        if not self.summary_checkpointer.checkpoint_exists(context.get_path(), data_container):
            return False

        current_ids = self.summary_checkpointer.read_summary(
            checkpoint_path=context.get_path(),
            data_container=data_container
        )

        for current_id in current_ids:
            if not self.data_input_checkpointer.checkpoint_exists(
                    checkpoint_path=self._get_data_input_checkpoint_path(context),
                    current_id=current_id
            ):
                return False

            if not self.expected_output_checkpointer.checkpoint_exists(
                    checkpoint_path=self._get_expected_output_checkpoint_path(context),
                    current_id=current_id
            ):
                return False

        return True

    def _get_data_input_checkpoint_path(self, context):
        return context.push(Identity(name=DataCheckpointType.DATA_INPUT.value)).get_path()

    def _get_expected_output_checkpoint_path(self, context):
        return context.push(Identity(name=DataCheckpointType.EXPECTED_OUTPUT.value)).get_path()


class DefaultCheckpoint(Checkpoint):
    """
    :class:`Checkpoint` with pickle mini data checkpointers wrapped in a :class:`MiniDataCheckpointerWrapper`, and the default step saving checkpointer.

    .. seealso::
        * :class:`Checkpoint`
        * :class:`MiniDataCheckpointerWrapper`
    """

    def __init__(self):
        Checkpoint.__init__(
            self,
            all_checkpointers=[
                StepSavingCheckpointer(),
                MiniDataCheckpointerWrapper(
                    summary_checkpointer=TextFileSummaryCheckpointer(),
                    data_input_checkpointer=PickleMiniDataCheckpointer(),
                    expected_output_checkpointer=PickleMiniDataCheckpointer()
                )
            ]
        )
