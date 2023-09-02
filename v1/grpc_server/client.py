import librosa 





from grpc_service import classifier_pb2
from grpc_service import classifier_pb2_grpc
import grpc

def run():
    with grpc.insecure_channel("localhost:5001") as channel:
        stub = classifier_pb2_grpc.SoundSourceClassifierStub(channel)

        
