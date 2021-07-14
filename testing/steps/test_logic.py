import numpy as np

from neuraxle.steps.logic import FilterColumns, Greater, Less, If


def test_filter():
    data_inputs = np.random.random((10, 2))
    filter = FilterColumns([
        (0, Greater(then=0.5)),
        (1, Less(then=0.5))
    ])

    outputs = filter.transform(data_inputs)

    for di in outputs:
        assert di[0] > 0.5
        assert di[1] < 0.5


def test_filter_if():
    data_inputs = np.random.random((10, 2))
    filter = FilterColumns([
        (0, If(condition_step=Greater(then=0.5), then_step=Greater(0.75), else_step=Less(0.25))),
    ])

    outputs = filter.transform(data_inputs)

    for di in outputs:
        assert di[0] > 0.75 or di[0] < 0.25
