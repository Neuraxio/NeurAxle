import numpy as np
from sklearn import linear_model

from neuraxle.distributed.clustering import ClusteringWrapper, RestWorker
from neuraxle.pipeline import Pipeline
from neuraxle.rest.flask import RequestWrapper
from neuraxle.steps.numpy import NumpyConcatenateOuterBatch

TIMESTEPS = 10
VALIDATION_SIZE = 0.1
BATCH_SIZE = 32
N_EPOCHS = 10

DATA_INPUTS_PAST_SHAPE = (BATCH_SIZE, TIMESTEPS)


class TestClientRequestWrapper(RequestWrapper):
    def __init__(self, test_client):
        self.test_client = test_client

    def post(self, url, files):
        files = ((name, content) for name, content in files.items())
        for name, content in files:
            r = self.test_client.post(url, data={'file': content}, content_type='multipart/form-data')
        return r

    def get(self, url, method, headers, data):
        r = self.test_client.get(url, data=data, headers=headers)
        return r.json


def test_clustering_wrapper(tmpdir):
    # Given
    app = RestWorker().get_app()
    app.config['UPLOAD_FOLDER'] = tmpdir
    app = app.test_client()
    app.post()
    data_inputs, expected_outputs = create_data()
    model_pipeline = Pipeline([linear_model.LinearRegression()])
    model_pipeline = model_pipeline.fit(data_inputs, expected_outputs)

    # When
    p = Pipeline([
        ClusteringWrapper(
            model_pipeline,
            hosts=['/pipeline'],
            joiner=NumpyConcatenateOuterBatch(),
            n_jobs=10,
            batch_size=10,
            n_workers=1,
            request=TestClientRequestWrapper(app)
        )
    ], cache_folder=tmpdir)
    outputs = p.transform(data_inputs)

    # Then
    assert len(outputs) == len(data_inputs)


def create_data():
    i = 0
    data_inputs = []
    for batch_index in range(BATCH_SIZE):
        batch = []
        for _ in range(TIMESTEPS):
            batch.append(i)
            i += 1
        data_inputs.append(batch)
    data_inputs = np.array(data_inputs)
    random_noise = np.random.random(DATA_INPUTS_PAST_SHAPE)
    expected_outputs = 3 * data_inputs + 4 * random_noise

    return data_inputs, expected_outputs
