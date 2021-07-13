"""
Tests for Hyperparameters Distribution Spaces
=============================================

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

import pytest

from neuraxle.hyperparams.distributions import *
from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples, RecursiveDict, \
    CompressedHyperparameterSamples

hyperparams_flat_and_dict_pairs = [
    # Pair 1:
    ({
         "a__learning_rate": 7
     },
     {
         "a": {
             "learning_rate": 7
         }
     }),
    # Pair 2:
    ({
         "b__a__learning_rate": 7,
         "b__learning_rate": 9
     },
     {
         "b": {
             "a": {
                 "learning_rate": 7
             },
             "learning_rate": 9
         }
     }),
]


@pytest.mark.parametrize("class_to_test", [RecursiveDict, HyperparameterSamples, HyperparameterSpace])
@pytest.mark.parametrize("flat,expected_dic", hyperparams_flat_and_dict_pairs)
def test_flat_to_dict_hyperparams(flat: dict, expected_dic: dict, class_to_test):
    from_flat_dic = class_to_test(flat)
    from_nested_dic = class_to_test(expected_dic)

    assert from_flat_dic == from_nested_dic
    assert from_flat_dic.to_flat_dict() == flat
    assert from_nested_dic.to_flat_dict() == flat
    assert from_nested_dic.to_nested_dict() == expected_dic
    assert from_flat_dic.to_nested_dict() == expected_dic


HYPE_SPACE = HyperparameterSpace({
    "a__test": Boolean(),
    "a__lr": Choice([0, 1, False, "Test"]),
    "a__b__c": PriorityChoice([0, 1, False, "Test"]),
    "a__b__q": Quantized(Uniform(-10, 10)),
    "d__param": RandInt(-10, 10),
    "d__u": Uniform(-10, 10),
    "e__other": LogUniform(0.001, 10),
    "e__alpha": Normal(0.0, 1.0),
    "e__f__g": LogNormal(0.0, 2.0),
    "p__other_nondistribution_params": "hey",
    "p__could_also_be_as_fixed": FixedHyperparameter("also hey"),
    "p__its_over_9k": 9001
})


def test_hyperparams_space_rvs_outputs_samples():
    space = copy.deepcopy(HYPE_SPACE)

    samples = space.rvs()

    assert isinstance(samples, HyperparameterSamples)
    assert len(samples) == len(space)
    for k, v in samples.iter_flat():
        assert k in space
        assert not isinstance(v, HyperparameterDistribution)


def test_hyperparameter_samples_compress():
    hps = HyperparameterSamples({
        "b__a__learning_rate": 7,
        "b__learning_rate": 9,
        "Sklearn__test1__test__abc": False,
        "Sklearn__test2__test__abc": "Parallel"
    }, separator="__")
    expected = [
        {'step_name': 'a', 'hyperparams': {'learning_rate': 7}, 'ancestor_steps': ['b']},
        {'step_name': 'b', 'hyperparams': {'learning_rate': 9}, 'ancestor_steps': []},
        {'step_name': 'test', 'hyperparams': {'abc': False}, 'ancestor_steps': ["Sklearn", "test1"]},
        {'step_name': 'test', 'hyperparams': {'abc': "Parallel"}, 'ancestor_steps': ["Sklearn", "test2"]}]

    actual = hps.compress()
    assert isinstance(actual, CompressedHyperparameterSamples)
    assert str(actual) == str(expected)


def test_hyperparameter_samples_compress_without_parents():
    hps = HyperparameterSamples({
        "b__a__learning_rate": 7,
        "b__learning_rate": 9,
        "Sklearn__test1__test__abc": False,
        "Sklearn__test2__test__abc": "Parallel"
    }, separator="__")
    expected = [
        {'step_name': 'a', 'hyperparams': {'learning_rate': 7}, 'ancestor_steps': None},
        {'step_name': 'b', 'hyperparams': {'learning_rate': 9}, 'ancestor_steps': None},
        {'step_name': 'test', 'hyperparams': {'abc': False}, 'ancestor_steps': None},
        {'step_name': 'test', 'hyperparams': {'abc': "Parallel"}, 'ancestor_steps': None}]
    actual = hps.compress(remove_parents=True)
    assert isinstance(actual, CompressedHyperparameterSamples)
    assert str(actual) == str(expected)


def test_hyperparameter_samples_compress_wildcards():
    hps = HyperparameterSamples({
        "b__a__learning_rate": 7,
        "b__learning_rate": 9,
        "Sklearn__test1__test__abc": False,
        "Sklearn__test2__test__abc": "Parallel"
    }, separator="__")
    expected = [('*a__learning_rate', 7), ('b__learning_rate', 9), ('*test1*abc', False),
                ("*test2*abc", 'Parallel')]
    actual = hps.compress().wildcards().items()
    assert list(actual) == expected
