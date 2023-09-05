import time
import json
import librosa
import numpy as np
import sounddevice as sd



from grpc_service import classifier_pb2
from grpc_service import classifier_pb2_grpc
import grpc


from tensorflow.keras.models import load_model


GENRES = [
        'disco',
        'blues',
        'rock',
        'pop',
        'metal',
        'country',
        'hiphop',
        'classical',
        'reggae'
        ]
model = load_model("genres_classification_cnn_v1.h5")



def run():
    with grpc.insecure_channel("localhost:5001") as channel:
        stub = classifier_pb2_grpc.SoundSourceClassifierStub(channel)

        settings_request = classifier_pb2.LibrosaSettingsRequest(settings_filename="appsettings.json")
        settings_reply = stub.LoadLibrosaSettings(settings_request)
        librosa_settings_dict = json.loads(settings_reply.librosa_settings_json.replace("'", '"'))

        sample_rate = librosa_settings_dict["sample_rate"]
        piece_len = librosa_settings_dict["piece_len"]
        wav_path = librosa_settings_dict["wav_path"]
        wav_len = librosa_settings_dict["wav_len"]
        n_fft = librosa_settings_dict["n_fft"]
        n_mfcc = librosa_settings_dict["n_mfcc"]
        hop_length = librosa_settings_dict["hop_length"]



        signal, sr = librosa.load(wav_path,  sr=sample_rate, offset=0, duration=piece_len)
# mode it on the server
        start_time = time.time()
        mfcc =  librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,
            n_fft=n_fft,
            n_mfcc=n_mfcc,
            hop_length=hop_length).T
        mfcc_arr = np.array(mfcc)[np.newaxis, ...]

        prediction = model.predict(mfcc_arr)
        max_arg_index = np.argmax(prediction, axis = 1)[0]
        genre = GENRES[max_arg_index]

        end_time = time.time()
        how_much_time = end_time - start_time
        print(str(how_much_time))
        print(genre)
        sd.play(signal, samplerate=sample_rate)






        pass




if __name__ == "__main__":
    run()