from model import LSTMmodel
from preparation_data import PreparationData

"""
класс PreparationData управляет подготовкой данных для нейросети
- он парсит данные
- нормализует их
- приводит в нужный формат

класс LSTMmodel
управляет моделью
- создает модель
- обучфет модель
- функционал тестирования

класс TestModel_and_ShowData строит график tm и состояний
- на графике я помножаю на 100 состояни чтобы было видно изменение


классы состояний
MOVE = 0 #движение 
BRAKING = 1 #начало торможение
REROOF = 2 #перекрыша
RELEASE = 3 #отпуск
STABILIZATION = 4 #стабилизация
"""

X, Y = PreparationData().data_education("1.json")
model = LSTMmodel("v2.keras")
model.test_model_evaluation(X, Y)




