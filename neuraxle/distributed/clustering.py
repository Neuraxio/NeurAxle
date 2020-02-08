"""
Neuraxle steps for distributed computing in the cloud
======================================================

Neuraxle Steps for distributed computing in the cloud

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
import os
from typing import List

import numpy as np
from flask import jsonify

from neuraxle.base import MetaStepMixin, BaseStep, NonFittableMixin, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.distributed.streaming import ParallelQueuedPipeline
from neuraxle.rest.flask import RestAPICaller, RequestWrapper, UrllibRequestWrapper
from neuraxle.steps.numpy import NumpyConcatenateInnerFeatures


class RestApiScheduler(ParallelQueuedPipeline):
    """
    Clustering scheduler that calls rest api endpoints for distributed processing.

    Please refer to the full example in :class:`ClusteringWrapper` for further details.

    .. seealso::
        :class:`BaseClusteringScheduler`,
        :class:`ClusteringWrapper`,
        :class:`ExecutionContext`,
        :class:`RestApiCaller`,
    """

    def __init__(
            self,
            hosts: List[str],
            batch_size,
            n_workers_per_step,
            max_size,
            data_joiner,
            request
    ):
        if request is None:
            request = UrllibRequestWrapper()
        self.request = request
        self.hosts = hosts

        step_name = 'rest'
        n_workers = len(hosts)
        additional_arguments = [('url', host) for host in hosts]
        step = ClusteringRestApiCaller(url=None, request=request)

        ParallelQueuedPipeline.__init__(
            self,
            steps=[(step_name, n_workers, additional_arguments, step)],
            batch_size=batch_size,
            n_workers_per_step=n_workers_per_step,
            max_size=max_size,
            data_joiner=data_joiner
        )

    def _will_process(self, data_container: DataContainer, context: ExecutionContext):
        """
        Spawn a worker for each host at the beginning of the processing.

        :param data_container:
        :param context:
        :return:
        """
        data_container, context = ParallelQueuedPipeline._will_process(self, data_container, context)
        self._spawn(context)

        return data_container, context

    def _did_process(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Kill all clusters at the end of the processing.

        :param data_container:
        :param context:
        :return:
        """
        data_container = ParallelQueuedPipeline._did_process(self, data_container, context)
        self._kill(context)

        return data_container

    def _kill(self, context: ExecutionContext):
        # TODO: what is there to kill here ?
        pass

    def _spawn(self, context: ExecutionContext):
        """
        Spawn a worker for each host.

        1) Send the full pipeline dump files to the workers using the rest api endpoints in self.hosts
        2) Add an :class:`ApiCaller` step in the queue of workers for each hosts.

        :param context: execution context
        :return:
        """
        self.started = True
        for host in self.hosts:
            saved_step_files = context.get_all_saved_step_files()
            files = {path.replace(str(context.root) + '/', ''): open(file_path, 'rb') for path, file_path in
                     saved_step_files}
            r = self.request.post(host, files=files)

            for f in files.values():
                f.close()

            if r.status_code != 201:
                raise Exception('Cannot Spawn Host: {}, status code: {}'.format(host, r.status_code))


class ClusteringWrapper(NonFittableMixin, MetaStepMixin, BaseStep):
    """
    Wrapper step for distributed processing. A :class:`BaseClusteringScheduler`, distributes the transform method to multiple workers.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            ClusteringWrapper(
                Pipeline([..]),
                hosts=['http://127.0.0.1:5000/pipeline'],
                joiner=NumpyConcatenateOuterBatch(),
                n_jobs=10,
                batch_size=10,
                n_workers_per_step=1,
                max_size=10
            )
        ])

    .. seealso::
        :class:`BaseClusteringScheduler`,
        :class:`LocalScheduler`,
        :class:`ExternalScheduler`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`,
        :class:`DataContainer`
    """

    def __init__(
            self,
            wrapped: BaseStep,
            hosts: List[str],
            joiner: NonFittableMixin = None,
            n_jobs: int = 10,
            batch_size=None,
            n_workers=None,
            max_size=None,
            request: RequestWrapper = None
    ):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        NonFittableMixin.__init__(self)

        self.scheduler = RestApiScheduler(
            hosts=hosts,
            batch_size=batch_size,
            n_workers_per_step=n_workers,
            max_size=max_size,
            data_joiner=joiner,
            request=request
        )

        if joiner is None:
            joiner = NumpyConcatenateInnerFeatures()
        self.joiner = joiner

        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def _transform_data_container(self, data_container, context):
        """
        Transform data container using distributed processing.

        :param data_container:
        :param context:
        :return:
        """
        wrapped_context = context.push(self.wrapped)
        wrapped_context = wrapped_context.push(wrapped_context.save_last())

        data_container = self.scheduler.handle_transform(data_container, wrapped_context)

        return data_container


class RestWorker:
    """
    Worker class for the :class:`RestApiScheduler`.


    Easily start a flask server for a rest api worker :

    .. code-block:: python

        worker = RestWorker(host='127.0.0.1', port=5000)
        worker.start(upload_folder='/home/neuraxle/Documents/cluster')


    .. seealso::
        :class:`BaseClusteringScheduler`,
        :class:`LocalScheduler`,
        :class:`ExternalScheduler`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`,
        :class:`DataContainer`
    """

    def __init__(self, ressource_name='pipeline', host: str = None, port: int = 5000):
        self.ressource_name = ressource_name
        self.host = host
        self.port = port

    def get_app(self):
        """
        :return: a Flask app (as given by `app = Flask(__name__)` and then configured).
        """
        from flask import Flask, request
        from flask_restful import Api, Resource

        app = Flask(__name__)
        api = Api(app)

        class PipelineRessource(Resource):
            def post(self):
                """
                Post route to save the full pipeline dump locally.
                Receives the full dump for each pipeline steps inside the files attribute.

                :return: response
                """
                if len(request.files) == 0:
                    response = jsonify({'message': 'No file part in the request'})
                    response.status_code = 400
                    return response

                for file, content in request.files.items():
                    filename = content.filename.split(os.path.sep)[-1]
                    upload_folder = str(app.config['UPLOAD_FOLDER'])
                    file_path = content.filename.replace(upload_folder, '').replace(filename, '')
                    if file_path[0] == os.sep:
                        file_path = file_path[1:]
                    upload_path = os.path.join(upload_folder, file_path)

                    if not os.path.exists(upload_path):
                        os.makedirs(upload_path)

                    file_path = os.path.join(upload_path, filename)
                    content.save(file_path)

                response = jsonify({'message': 'Files successfully uploaded'})
                response.status_code = 201

                return response

            def get(self):
                """
                Get route to transform the data container received in the body :

                .. code-block:: python

                    {
                        current_ids: [...],
                        data_inputs: [...],
                        expected_outputs: [...],
                        summary_id: ...
                    }

                :return: transformed data container
                """
                context = ExecutionContext(app.config['UPLOAD_FOLDER'])
                body = request.get_json()
                pipeline = context.load(body['path'])

                data_container = pipeline.handle_transform(
                    DataContainer(
                        current_ids=body['current_ids'],
                        data_inputs=body['data_inputs'],
                        expected_outputs=body['expected_outputs'],
                        summary_id=body['summary_id'],
                    ),
                    context
                )

                response = jsonify({
                    'current_ids': np.array(data_container.current_ids).tolist(),
                    'data_inputs': np.array(data_container.data_inputs).tolist(),
                    'expected_outputs': np.array(data_container.expected_outputs).tolist(),
                    'summary_id': data_container.summary_id
                })
                response.status_code = 200

                return response

        api.add_resource(PipelineRessource, '/' + self.ressource_name)

        return app

    def start(self, upload_folder):
        """
        Start the worker with the given upload folder.

        :param upload_folder: upload folder
        :type upload_folder: str
        :return:
        """
        app = self.get_app()
        app.config['UPLOAD_FOLDER'] = upload_folder
        app.run(host=self.host, debug=False, port=self.port)


class ClusteringRestApiCaller(RestAPICaller):
    def _will_process(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        data_container, context = RestAPICaller._will_process(self, data_container, context)

        # remove QueuedWorker from the context
        context.pop()

        return data_container, context
