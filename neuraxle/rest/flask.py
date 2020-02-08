"""
Neuraxle's Flask Wrapper classes
====================================
The flask wrapper classes are used to easily serve pipeline predictions using a flask rest api.

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
import json
import urllib
from abc import ABC, abstractmethod
from urllib import request

import numpy as np
from flask import Response

from neuraxle.base import BaseStep, NonFittableMixin, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.pipeline import Pipeline


class JSONDataBodyDecoder(NonFittableMixin, BaseStep, ABC):
    """
    Class to be used within a FlaskRESTApiWrapper to convert input json to actual data (e.g.: arrays)
    """

    def transform(self, data_inputs):
        return self.decode(data_inputs)

    @abstractmethod
    def decode(self, data_inputs: dict):
        """
        Will convert data_inputs to a dict or a compatible data structure for jsonification

        :param encoded data_inputs (dict parsed from json):
        :return: data_inputs (as a data structure compatible with pipeline's data inputs)
        """
        raise NotImplementedError("TODO: inherit from the `JSONDataBodyDecoder` class and implement this method.")


class JSONDataResponseEncoder(NonFittableMixin, BaseStep, ABC):
    """
    Base class to be used within a FlaskRESTApiWrapper to convert prediction output to json response.
    """

    def transform(self, data_inputs) -> Response:
        """
        Transform processed data inputs into a flask response object.

        :param data_inputs:
        :return: flask response object
        """
        from flask import jsonify
        return jsonify(self.encode(data_inputs))

    @abstractmethod
    def encode(self, data_inputs) -> dict:
        """
        Convert data_inputs to a dict or a compatible data structure for jsonification.

        :param data_inputs (a data structure outputted by the pipeline after a transform):
        :return: encoded data_inputs (jsonifiable dict)
        """
        raise NotImplementedError("TODO: inherit from the `JSONDataResponseEncoder` class and implement this method.")


class FlaskRestApiWrapper(Pipeline):
    """
    Wrap a pipeline to easily deploy it to a REST API. Just provide a json encoder and a json decoder.

    Usage example:

    ```
    class CustomJSONDecoderFor2DArray(JSONDataBodyDecoder):
        '''This is a custom JSON decoder class that precedes the pipeline's transformation.'''

        def decode(self, data_inputs: dict):
            values_in_json_2d_arr: List[List[int]] = data_inputs["values"]
            return np.array(values_in_json_2d_arr)

    class CustomJSONEncoderOfOutputs(JSONDataResponseEncoder):
        '''This is a custom JSON response encoder class for converting the pipeline's transformation outputs.'''

        def encode(self, data_inputs) -> dict:
            return {
                'predictions': list(data_inputs)
            }

    app = FlaskRestApiWrapper(
        json_decoder=CustomJSONDecoderFor2DArray(),
        wrapped=Pipeline(...),
        json_encoder=CustomJSONEncoderOfOutputs(),
    ).get_app()

    app.run(debug=False, port=5000)
    ```
    """

    def __init__(
            self,
            json_decoder: JSONDataBodyDecoder,
            wrapped: BaseStep,
            json_encoder: JSONDataResponseEncoder,
            route='/'):
        Pipeline.__init__(self, [
            json_decoder,
            wrapped,
            json_encoder
        ])
        self.route: str = route

    def get_app(self):
        """
        This methods returns a REST API wrapping the pipeline.

        :return: a Flask app (as given by `app = Flask(__name__)` and then configured).
        """
        from flask import Flask, request
        from flask_restful import Api, Resource

        app = Flask(__name__)
        api = Api(app)
        wrapped = self

        class RESTfulRes(Resource):
            def get(self):
                return wrapped.transform(request.get_json())

        api.add_resource(
            RESTfulRes,
            self.route
        )

        return app


class RestAPICaller(NonFittableMixin, BaseStep):
    def __init__(self, url, method='GET', request= None):
        BaseStep.__init__(self)
        self.url = url
        self.method = method
        if request is None:
            request = urllib.request
        self.request = request

    def transform(self, data_inputs):
        raise NotImplementedError('must be used inside a pipeline')

    def _will_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        return data_container, context

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        data_dict = {
            'path': context.get_path(False),
            'data_inputs': np.array(data_container.data_inputs).tolist(),
            'expected_outputs': np.array(data_container.expected_outputs).tolist(),
            'current_ids': np.array(data_container.current_ids).tolist(),
            'summary_id': data_container.summary_id
        }
        data = json.dumps(data_dict).encode('utf8')

        results = self.request.get(
            self.url,
            method=self.method,
            headers={'content-type': 'application/json'},
            data=data
        )

        return DataContainer(
            summary_id=np.array(results['summary_id']),
            current_ids=np.array(results['current_ids']),
            data_inputs=np.array(results['data_inputs']),
            expected_outputs=np.array(results['expected_outputs'])
        )


class RequestWrapper(ABC):
    @abstractmethod
    def post(self, host, files):
        pass

    @abstractmethod
    def get(self, url, method, headers, data):
        pass


class UrllibRequestWrapper(RequestWrapper):
    def __init__(self):
        pass

    def post(self, url, files):
        pass

    def get(self, url, method, headers, data):
        req = request.Request(url, method=method, headers=headers, data=data)
        response = request.urlopen(req)
        results = json.loads(response.read())

        return results