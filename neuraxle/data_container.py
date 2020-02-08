"""
Neuraxle's DataContainer classes
====================================
Classes for containing the data that flows throught the pipeline steps.

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
import math
from typing import Any, Iterable, List

from conv import convolved_1d


class DataContainer:
    """
    DataContainer class to store data inputs, expected outputs, and ids together.
    Each :class:`BaseStep` needs to rehash ids with hyperparameters so that the :class:`Checkpoint` step
    can create checkpoints for a set of hyperparameters.

    The DataContainer object is passed to all of the :class:`BaseStep` handle methods :
        * :func:`~neuraxle.base.BaseStep.handle_transform`
        * :func:`~neuraxle.base.BaseStep.handle_fit_transform`
        * :func:`~neuraxle.base.BaseStep.handle_fit`

    Most of the time, you won't need to care about the DataContainer because it is the pipeline that manages it.

    .. seealso::
        :class:`BaseHasher`,
        :class: `BaseStep`
    """

    def __init__(self, data_inputs: Any, current_ids=None, summary_id=None, expected_outputs: Any = None):
        self.current_ids = current_ids
        self.summary_id = summary_id

        self.data_inputs = data_inputs
        if expected_outputs is None and isinstance(data_inputs, Iterable):
            self.expected_outputs = [None] * len(data_inputs)
        else:
            self.expected_outputs = expected_outputs

    def set_data_inputs(self, data_inputs: Iterable):
        """
        Set data inputs.

        :param data_inputs: data inputs
        :type data_inputs: Iterable
        :return:
        """
        self.data_inputs = data_inputs
        return self

    def set_expected_outputs(self, expected_outputs: Iterable):
        """
        Set expected outputs.

        :param expected_outputs: expected outputs
        :type expected_outputs: Iterable
        :return:
        """
        self.expected_outputs = expected_outputs
        return self

    def set_current_ids(self, current_ids: List[str]):
        """
        Set current ids.

        :param current_ids: data inputs
        :type current_ids: List[str]
        :return:
        """
        self.current_ids = current_ids
        return self

    def set_summary_id(self, summary_id: str):
        """
        Set summary id.

        :param summary_id: str
        :return:
        """
        self.summary_id = summary_id
        return self

    def hash_summary(self):
        """
        Hash :class:`DataContainer`.current_ids, data inputs, and hyperparameters together into one id.

        :return: single hashed current id for all of the current ids
        :rtype: str
        """
        m = hashlib.md5()
        for current_id in self.current_ids:
            m.update(str.encode(str(current_id)))
        return m.hexdigest()

    def convolved_1d(self, stride, kernel_size) -> Iterable['DataContainer']:
        """
        Returns an iterator that iterates through batches of the DataContainer.

        :param stride: step size for the convolution operation
        :param kernel_size:
        :return: an iterator of DataContainer
        :rtype: Iterable[DataContainer]

        .. seealso::
            `<https://github.com/guillaume-chevalier/python-conv-lib>`_
        """
        conv_current_ids = convolved_1d(stride=stride, iterable=self.current_ids, kernel_size=kernel_size,
                                        include_incomplete_pass=True)
        conv_data_inputs = convolved_1d(stride=stride, iterable=self.data_inputs, kernel_size=kernel_size,
                                        include_incomplete_pass=True)
        conv_expected_outputs = convolved_1d(stride=stride, iterable=self.expected_outputs, kernel_size=kernel_size,
                                             include_incomplete_pass=True)

        for current_ids, data_inputs, expected_outputs in zip(conv_current_ids, conv_data_inputs,
                                                              conv_expected_outputs):
            for i, (ci, di, eo) in enumerate(zip(current_ids, data_inputs, expected_outputs)):
                if di is None:
                    current_ids = current_ids[:i]
                    data_inputs = data_inputs[:i]
                    expected_outputs = expected_outputs[:i]
                    break

            yield DataContainer(data_inputs=data_inputs, current_ids=current_ids, summary_id=self.summary_id,
                                expected_outputs=expected_outputs)

    def get_n_batches(self, batch_size: int) -> int:
        return math.ceil(len(self.data_inputs) / batch_size)

    def get_n_baches(self, batch_size: int) -> int:
        return math.ceil(self.data_inputs / batch_size)

    def copy(self):
        return DataContainer(data_inputs=self.data_inputs, current_ids=self.current_ids, summary_id=self.summary_id,
                             expected_outputs=self.expected_outputs)

    def __iter__(self):
        """
        Iter method returns a zip of all of the current_ids, data_inputs, and expected_outputs in the data container.

        :return: iterator of tuples containing current_ids, data_inputs, and expected outputs
        :rtype: Iterator[Tuple]
        """
        current_ids = self.current_ids
        if self.current_ids is None:
            current_ids = [None] * len(self.data_inputs)

        expected_outputs = self.expected_outputs
        if self.expected_outputs is None:
            expected_outputs = [None] * len(self.data_inputs)

        return zip(current_ids, self.data_inputs, expected_outputs)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__ + "(current_ids=" + repr(list(self.current_ids)) + ", summary_id=" + repr(
            self.summary_id)

    def __len__(self):
        return len(self.data_inputs)


class ExpandedDataContainer(DataContainer):
    """
    Sub class of DataContainer to expand data container dimension.

    .. seealso::
        :class:`ExpandedDataContainer`,
    """

    def __init__(self,  data_inputs, current_ids, expected_outputs, summary_id, old_current_ids):

        DataContainer.__init__(self, data_inputs=data_inputs, current_ids=current_ids, summary_id=summary_id,
                               expected_outputs=expected_outputs)

        self.old_current_ids = old_current_ids

    def reduce_dim(self) -> 'DataContainer':
        """
        Reduce DataContainer to its original shape with a list of multiple current_ids, data_inputs, and expected outputs.

        :return: reduced data container
        :rtype: DataContainer
        """
        return DataContainer(data_inputs=self.data_inputs[0], current_ids=self.old_current_ids,
                             summary_id=self.summary_id, expected_outputs=self.expected_outputs[0])

    @staticmethod
    def create_from(data_container: DataContainer) -> 'ExpandedDataContainer':
        """
        Create ExpandedDataContainer with the given summary hash for the single current id.

        :param data_container: data container to transform
        :type data_container: DataContainer
        :return: expanded data container
        :rtype: ExpandedDataContainer
        """
        return ExpandedDataContainer(data_inputs=[data_container.data_inputs], current_ids=[data_container.summary_id],
                                     summary_id=data_container.summary_id,
                                     expected_outputs=[data_container.expected_outputs], old_current_ids=data_container.current_ids)


class ListDataContainer(DataContainer):
    """
    Sub class of DataContainer to perform list operations.
    It allows to perform append, and concat operations on a DataContainer.

    .. seealso::
        :class:`DataContainer`
    """

    @staticmethod
    def empty() -> 'ListDataContainer':
        return ListDataContainer([], [], [])

    def append(self, current_id: str, data_input: Any, expected_output: Any):
        """
        Append a new data input to the DataContainer.

        :param current_id: current id for the data input
        :type current_id: str
        :param data_input: data input
        :param expected_output: expected output
        :return:
        """
        self.current_ids.append(current_id)
        self.data_inputs.append(data_input)
        self.expected_outputs.append(expected_output)

    def append_data_container(self, data_container: DataContainer):
        self.data_inputs.append(data_container)

    def concat(self, data_container: DataContainer):
        """
        Concat the given data container to the current data container.

        :param data_container: data container
        :type data_container: DataContainer
        :return:
        """
        self.current_ids.extend(data_container.current_ids)
        self.data_inputs.extend(data_container.data_inputs)
        self.expected_outputs.extend(data_container.expected_outputs)
