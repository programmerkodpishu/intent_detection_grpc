import librosa

from grpc_service import classifier_pb2
from grpc_service import classifier_pb2_grpc
import grpc

def run(signal, sr, model_name, librosa_settings):
    with grpc.insecure_channel("localhost:5001") as channel:
        stub = classifier_pb2_grpc.SoundSourceClassifierStub(channel)
        get_music_genre_request = classifier_pb2.MusicGenreRequest(
            signal=signal.tolist(),
            sample_rate=sr,
            model_name=model_name,
            librosa_settings=str(librosa_settings))

        music_genre_response = stub.PredictMusicGenre(get_music_genre_request)
        pass


if __name__ == "__main__":
    librosa_settings = {
        "sample_rate": 8000,
        "duration": 0.3,

        "n_fft": 256,
        "n_mfcc": 13,
        "hop_length": 64,

    }
    WAV_PATH = "9_genres_record.wav"
    model_name = "genres_classification_cnn_v1.h5"

    signal, sr = librosa.load(WAV_PATH,
                              sr=librosa_settings["sample_rate"],
                              offset=0,
                              duration=librosa_settings["duration"])

    run(signal=signal, sr=sr, model_name=model_name, librosa_settings=librosa_settings)
