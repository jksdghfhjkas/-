import json
import numpy as np
from numpy import ndarray
from keras.utils import to_categorical
from os.path import exists

class PreparationData:

    def __dp_or_ddp_data(self, data: ndarray):
        """
        выдает первую и вторую производную
        """
        dp = np.diff(data, prepend=data[0])
        ddp = np.diff(dp, prepend=dp[0])
        return dp, ddp
    

    def __normalized_data(self, data: list[ndarray], filename):
        """
        нормализую данные через StandardScaler
        z = (x - u)/q
        u - среднее
        q - отклонение
        data это tm dp ddp
        """

        def get_u_q(data: ndarray):
            u = np.mean(data) 
            q = np.std(data, ddof=0)
            return u, q
        

        if not exists(f"normalization/{filename}"):
            # если файла с параметрами нет то мы их рассчитываем
            u, q = get_u_q(data[0])
            u1, q1 = get_u_q(data[1])
            u2, q2 = get_u_q(data[2])

            with open(f"normalization/{filename}", "w+") as file:
                json_data = {
                    "tm": [u, q],
                    "dp": [u1, q1],
                    "ddp": [u2, q2]
                }
                json.dump(json_data, file)

        else:
            # если есть то просто берем
            with open(f"normalization/{filename}", "r") as file:
                json_data = json.load(file)
                u, q = json_data.get("tm")
                u1, q1 = json_data.get("dp")
                u2, q2 = json_data.get("ddp")
            

        z = (data[0] - u) / q
        z1 = (data[1] - u1) / q1
        z2 = (data[2] - u2) / q2
                
        return z, z1, z2


    def __parsing_file(self, filename: str):
        with open(f"education_data/{filename}", "r") as file:
            data = json.load(file)
            # pm = data.get("pm") 
            # yp = data.get("yp")
            tm = data.get("tm")

            states = data.get("states")

        return tm, states
    
    
    def data_push_batch(self, tm: list | ndarray, filename: str) -> ndarray:
        """
        получает на вход окно, нормализует и формирует данные
        filename - это файл с параметрами нормализаций
        """
        if len(tm) != 5:
            raise ValueError("Неверная длина данных")
        
        tm = np.array(tm)
        dp, ddp = self.__dp_or_ddp_data(tm)

        tm, dp, ddp = self.__normalized_data([tm, dp, ddp], filename)

        result = np.column_stack([tm, dp, ddp])
        result = np.expand_dims(result, axis=0)

        return np.array(result)
    
    
    def data_push(self, tm: list | ndarray, filename: str) -> ndarray:
        """
        работает со всеми данными полученными
        filename - это файл с параметрами нормализаций
        """

        tm = np.array(tm)
        dp, ddp = self.__dp_or_ddp_data(tm)
        tm, dp, ddp = self.__normalized_data([tm, dp, ddp], filename)

        result_x = []
        lenght = []

        # разделяем на пачки по 5
        for i, v in enumerate(tm):
            
            if len(lenght) == 5:
                time = np.array(lenght.copy())
                result_x.append(time)
                lenght.pop(0)

            lenght.append(np.array([v, dp[i], ddp[i]]))

        return np.array(result_x)


    def data_education(self, filename: str):
        """
        возврящает пачки по 5 элементов для обучения
        filename - это подготовленный файл с данными для обучения
        """
        tm, states = self.__parsing_file(filename)
        tm = np.array(tm)

        dp, ddp = self.__dp_or_ddp_data(tm)
        tm, dp, ddp = self.__normalized_data([tm, dp, ddp], filename)

        states = to_categorical(states, num_classes=5)

        result_x = []
        result_y = []
        lenght = []


        # разделяем на пачки по 5
        for i, v in enumerate(tm):
            
            if len(lenght) == 5:
                time = np.array(lenght.copy())
                result_x.append(time)
                result_y.append(np.array(states[i - 1]))
                lenght.pop(0)

            lenght.append(np.array([v, dp[i], ddp[i]]))

        return np.array(result_x), np.array(result_y)

    

        

        