import json
from concurrent import futures

import grpc 
from grpc_service import classifier_pb2
from grpc_service import classifier_pb2_grpc




class SoundClassifierService(classifier_pb2_grpc.SoundSourceClassifierServicer):
    def __init__(self):
        self.settings = self.load_settings()


    def PredictMusicGenre(self, request, context):
        return super().PredictMusicGenre(request, context)
    

    def load_settings(self):
        with open("appsettings.json") as fp:
            return json.load(fp)
            

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    classifier_pb2_grpc.add_SoundSourceClassifierServicer_to_server(SoundClassifierService(), server)
    server.add_insecure_port("localhost:5001")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()