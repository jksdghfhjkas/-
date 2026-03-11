import matplotlib.pyplot as plt
import pandas
from model import LSTMmodel
from preparation_data import PreparationData
import numpy as np



"""
тут просто отрисовка работы нейросети на графике

он рисует график ответа нейросети
"""

class TestModel_and_ShowData:
    def __init__(self, filename):
        self.filename = filename
        self.model = LSTMmodel("v2.keras")
        self.preparation_data = PreparationData()
        

        self.tm = self.__parsing_file()
        self.states = self.__get_states()
        

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.set_xlabel('Время * 0.2')
        self.ax.set_ylabel('Значение')
        self.ax.grid(True)


    def __parsing_file(self):
        data_set = pandas.read_excel(f"excel_data/{self.filename}", usecols="A:D")

        data_set.columns = ["A", "B", "C", "D"]
                    
        print("Файл успешно прочитан")

        condition = (data_set["B"] != -1) | (data_set["C"] != -1) | (data_set["D"] != -1)
        filtered = data_set[condition]
                    
        data = [filtered[col].tolist() for col in filtered.columns]

        return data[1]


    def __get_states(self):
        """
        тут мы данные отправляет на классификацию нейронке
        """

        X = self.preparation_data.data_push(self.tm, "1.json")
        states = self.model.test_model_states(X)
        return states


    def show(self):
        if not plt.fignum_exists(self.fig.number):
            raise Exception("Окно закрыто, прекращаем выполнение")

        # чтобы было видно изменнения на графике
        states = np.array(self.states) * 100

        if len(states) != len(self.tm):
            min_len = min(len(states), len(self.tm))
            self.tm = self.tm[:min_len]
            states = states[:min_len]

        self.ax.plot(range(len(self.tm)), self.tm, "r-", linewidth=2, label="TM")
        self.ax.plot(range(len(self.tm)), states, "v-", linewidth=2, label="states")
        plt.show()



if __name__ == "__main__":
    show = TestModel_and_ShowData("торможение.xlsx")
    show.show()