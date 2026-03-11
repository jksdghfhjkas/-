import keras
from keras.layers import Input, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from keras.models import load_model
import numpy as np
from numpy import ndarray


class LSTMmodel:
    def __init__(self, model: str | None = None):
        self.batch_size = 50
        self.epochs = 50

        if model is None:
            self.model = self.__build_model()
        else:
            self.model = load_model(f"save_models/{model}")

    
    def __build_model(self):
        inputs = Input(shape=(5, 3))

        x = LSTM(
            64, 
            return_sequences=True, 
            name="lstm_1"
        )(inputs)

        x = LSTM(
            32, 
            return_sequences=False, 
            name="lstm_2"
        )(x)

        x = Dense(
            16, 
            activation="relu",
            name="dense_1"
        )(x)

        x = Dropout(
            0.3,
            name="dropout_1"
        )(x)

        output = Dense(5, activation="softmax", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=output)

        model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss='categorical_crossentropy',
            metrics=[],
            weighted_metrics=['accuracy']
            
        )

        return model
    

    def push_data(self, data: ndarray) -> ndarray:
        """
        отправка данных
        """
        result = self.model.predict(data, verbose=0)
        state = np.argmax(result[0][0])
        return state
    
    
    def test_model_states(self, X: ndarray) -> ndarray:
        """
        прогоняем данные и получаем массив состояний
        """
        states = self.model.predict(X)
        states = np.argmax(states, axis=1)
        return states



    def test_model_evaluation(self, X: ndarray, Y: ndarray):
        """
        просто прогонка тестовых данных для проверки модели
        """
        evaluation = self.model.evaluate(X, Y, verbose=0)
        print(str(evaluation))


    def education(self, X: ndarray, Y: ndarray, filename: str = "v3"):

        class_weights = {
            0: 0.5,  # Движение - маленькая важность
            1: 2.0,  # Торможение - высокая важность
            2: 2.0,  # Перекрышка - высокая важность
            3: 2.0,  # Отпуск - высокая важность
            4: 2.0   # Стабилизация - высокая важность
        }

        history = self.model.fit(
            x=X,
            y=Y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            shuffle=False,
            class_weight=class_weights,
        )

        evaluation = self.model.evaluate(X, Y, verbose=0)

        print(str(evaluation))

        self.model.save(f"save_models/{filename}.keras")
        return history
    



        
