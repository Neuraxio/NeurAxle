import numpy as np

from neuraxle.base import BaseStep, ExecutionContext, NonFittableMixin
from neuraxle.data_container import DataContainer, ListDataContainer
from neuraxle.hyperparams.distributions import HyperparameterDistribution, FixedHyperparameter
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.pipeline import Pipeline
from neuraxle.steps.column_transformer import ColumnChooserTupleList, ColumnsSelectorND


class FilterColumns(NonFittableMixin, Pipeline):
    def __init__(self, conditions: ColumnChooserTupleList, n_dimension: int = 3):
        BaseStep.__init__(self)
        self.conditions = conditions
        self.string_indices = [
            str(name) + "_" + str(step.__class__.__name__)
            for name, step in conditions
        ]

        Pipeline.__init__(self, [
            (string_indices, Pipeline([
                ColumnsSelectorND(indices, n_dimension=n_dimension),
                step
            ]))
            for string_indices, (indices, step) in zip(self.string_indices, conditions)
        ])

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        filtered = ListDataContainer.empty()

        for current_id, data_input, expected_output in data_container:
            if np.all([step.transform([data_input]) for _, step in self]):
                filtered.append(current_id, data_input, expected_output)

        return filtered


class If(NonFittableMixin, Pipeline):
    def __init__(self, condition_step: BaseStep, then_step: BaseStep, else_step: BaseStep):
        NonFittableMixin.__init__(self)
        Pipeline.__init__(self, [
            ('condition_step', condition_step),
            ('then_step', then_step),
            ('else_step', else_step)
        ])

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        if self['condition_step'].transform(data_container.data_inputs):
            data_inputs = self['then_step'].transform(data_container.data_inputs)
        else:
            data_inputs = self['else_step'].transform(data_container.data_inputs)

        data_container.set_data_inputs(data_inputs)

        return data_container


class GreaterOrEqual(NonFittableMixin, BaseStep):
    def __init__(self, then, space: HyperparameterDistribution = None):
        if space is None:
            space = FixedHyperparameter(then)
        BaseStep.__init__(
            self,
            hyperparams=HyperparameterSamples({'then': then}),
            hyperparams_space=HyperparameterSpace({'then': space})
        )
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        return data_inputs >= self.hyperparams['then']


class LessOrEqual(NonFittableMixin, BaseStep):
    def __init__(self, then, space: HyperparameterDistribution = None):
        if space is None:
            space = FixedHyperparameter(then)
        BaseStep.__init__(
            self,
            hyperparams=HyperparameterSamples({'then': then}),
            hyperparams_space=HyperparameterSpace({'then': space})
        )
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        return data_inputs <= self.hyperparams['then']


class Greater(NonFittableMixin, BaseStep):
    def __init__(self, then, space: HyperparameterDistribution = None):
        if space is None:
            space = FixedHyperparameter(then)

        BaseStep.__init__(
            self,
            hyperparams=HyperparameterSamples({'then': then}),
            hyperparams_space=HyperparameterSpace({'then': space})
        )
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        return data_inputs > self.hyperparams['then']


class Less(NonFittableMixin, BaseStep):
    def __init__(self, then, space: HyperparameterDistribution = None):
        if space is None:
            space = FixedHyperparameter(then)
        BaseStep.__init__(
            self,
            hyperparams=HyperparameterSamples({'then': then}),
            hyperparams_space=HyperparameterSpace({'then': space})
        )
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        return data_inputs < self.hyperparams['then']


class TrueStep(NonFittableMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        return True


class FalseStep(NonFittableMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        return False
