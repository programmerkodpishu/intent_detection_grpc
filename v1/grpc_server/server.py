import json
import time
from concurrent import futures
import librosa
import numpy as np

import tensorflow as tf
import os
import sys

sys.path.append(os.getcwd())

import grpc
from grpc_service import classifier_pb2
from grpc_service import classifier_pb2_grpc

'''
Последовательность загрузки сервера

1. Загрузка модели 
2. Загрузка настроек
3. Получение кусочка аудиозаписи
4. Определение источника звука
5. Ответ строкой


Входные данные.
1. 10_000 пользователелей и у всех разные запросы
2. Каждый пользователь может хотеть распознавать какие-то отедльные звуки -> нужны разные модели
3. Каждый пользователь будет хотеть использовать разные настройки для либрозы
4. Каждый пользователь будет хотеть использовать какие-то иные модели.
5. Каждый пользователь будет посылать аудиодорожку какий-то нефиксированной длинны 


Значит
В каждом запросе на распознование будет необходимо присылать
1. Массив значений сигнала
2. Сэмпл рейт
3. Какую модель хочет использовать (для распознавания жанра музыки, шумы улицы, мужской или женский голос) -> нужно дообучить модели.  
4. Какие настройки для либрозы нужно использовать ( свои собственные или какой-то готовый пресет) 


def do_job(
    signal = signal,
    sr=sr,
    
    model="music_genres_classification_model",
    librosa_settings=librosa_settings,
    ) -> ["rock", "blues", "country", ... "genre"]
    
    1. Проверяем соответствует ли sr sr из librosa_settings
    2. Дробим сигнал по sr
    3. Определяем модель по названию модели и настройками 
    4. решаем задачу классификации
    5. Отдаём ответ
    
    
    
    


models = 
'''

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


class SoundClassifierService(classifier_pb2_grpc.SoundSourceClassifierServicer):
    def __init__(self, model_librosa_settings_json):
        self.models = self.upload_models(model_librosa_settings_json)

    @staticmethod
    def upload_models(self, model_librosa_settings_json):
        path = f"{os.getcwd()}/models/{model_librosa_settings_json}"
        with open(path, 'r') as fp:
            settings = json.load(fp)
            r = dict()
            for s in settings:
                tmp = dict()
                model_name = s["model_name"]
                try:
                    model = tf.keras.models.load_model(f"{os.getcwd()}/{model_name}")
                    tmp["model"] = model
                    tmp["is_ready"] = True

                except Exception as ex:
                    tmp["exception"] = ex
                    tmp["is_ready"] = False

                for key, item in s.items():
                    tmp[key] = item

                r[model_name] = tmp

            return r

    def get_model(self, reqeust):

        if not reqeust.model_name in self.models.keys():
            return

    def PredictMusicGenre(self, request, context):

        start_time = time.time()

        signal_ln = len(request.signal)
        if signal_ln == 0:
            end_time = time.time()
            return classifier_pb2.MusicGenreResponse(
                genres=[],
                status="signal len is 0",
                exception=str(None),
                execution_time=end_time - start_time)

        if not request.model_name in self.models.keys():
            end_time = time.time()
            return classifier_pb2.MusicGenreResponse(
                genres=[],
                status="no_such_model",
                exception=str(None),
                execution_time=end_time - start_time)

        librosa_settings = json.loads(request.librosa_settings.replace("'", '"'))

        model: tf.keras.Model = tf.keras.Model(self.models[request.model_name])
        requested_sr = librosa_settings["sample_rate"]
        model_sr = model["sample_rate"]

        if requested_sr != model_sr:
            end_time = time.time()

            return classifier_pb2.MusicGenreResponse(

                genres=[],
                status=f"different_sample_rates. Expected {model_sr} requested {requested_sr} ",
                exception=str(None),
                execution_time=end_time - start_time)

        sub_signals = np.array(request.signal)[::model_sr]

        r = []

        for s in sub_signals:
            mfcc = librosa.feature.mfcc(
                y=s,
                sr=librosa_settings["sample_rate"],
                n_fft=librosa_settings["n_fft"],
                n_mfcc=librosa_settings["n_mfcc"],
                hop_lengt=librosa_settings["hop_length"]
            ).T
            mfcc_arr = np.array(mfcc)[np.newaxis, ...]

            prediction = model.predict(mfcc_arr)
            max_arg = np.argmax(prediction, axis=1)[0]
            genre = GENRES[max_arg]
            r.append(genre)

        end_time = time.time()

        return classifier_pb2.MusicGenreResponse(

            genres=r,
            status=f"done",
            exception=str(None),
            execution_time=end_time - start_time)




def serve(model_librosa_settings_json):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = SoundClassifierService(

    )

    classifier_pb2_grpc.add_SoundSourceClassifierServicer_to_server(service, server)
    server.add_insecure_port("localhost:5001")
    print("starting")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve(
        model_librosa_settings_json="models_librosa_settings.json"
    )
    print("finished")
