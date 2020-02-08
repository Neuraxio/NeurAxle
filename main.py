from neuraxle.distributed.clustering import RestWorker

if __name__ == '__main__':
    worker = RestWorker(port=5000)
    worker.start(upload_folder='/home/alexandre/Documents/cluster')
    while True:
        pass
