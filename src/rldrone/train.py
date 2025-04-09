import gymnasium  # Библиотека для создания сред RL (аналог OpenAI Gym)
import os  # Для работы с файловой системой
import random  # Для генерации случайных ID
import string  # Для работы со строками
import argparse  # Для обработки аргументов командной строки

from PyFlyt.gym_envs import FlattenWaypointEnv  # Среда для управления квадрокоптером
from stable_baselines3.common.env_util import make_vec_env  # Создание векторной среды (для параллельных симуляций)
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC  # Алгоритмы DRL

def train(args):
    environment = args['environment']

    #Выбор алгоритма DRL, здесь выбирается алгоритм обучения (по умолчанию — PPO).
    if args['algorithm'] == "PPO": algorithm = PPO
    elif args['algorithm'] == "A2C": algorithm = A2C
    elif args['algorithm'] == "DDPG": algorithm = DDPG
    elif args['algorithm'] == "TD3": algorithm = TD3
    elif args['algorithm'] == "SAC": algorithm = SAC
    else:
        print("Error: Invalid DRL Algorithm specified")
        return
    
    #Генерация уникального ID для эксперимента
    id = ''.join(random.choices(string.ascii_letters, k=20))
    full_id = args['algorithm'] + '_' + environment + '_' + id

    #Создание папок для моделей и логов
    models_dir = f"models/{full_id}"
    logdir = "data"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    #Создание среды обучения, cоздаётся случайный 20-символьный ID, чтобы различать разные запуски
    train_env = make_vec_env(lambda: FlattenWaypointEnv(gymnasium.make("PyFlyt/QuadX-Waypoints-v4"), context_length=1), n_envs=1)
    """
    PyFlyt/QuadX-Waypoints-v4 — среда для управления квадрокоптером
    FlattenWaypointEnv — обёртка, которая упрощает наблюдения (observations)
    make_vec_env — создаёт векторную среду (можно обучать на нескольких параллельных симуляциях)
    """
    #Инициализация модели DRL
    model = algorithm("MlpPolicy", train_env, verbose=1, tensorboard_log=logdir if args["log"] else None)
    """
        "MlpPolicy" — нейросеть с полносвязными слоями (MLP).
        tensorboard_log=logdir — логирование в TensorBoard (если args["log"] == True).
    """

    # Обучение модели
    for i in range(1, args['num_iters']):
        model.learn(total_timesteps=args['steps_per_iter'], reset_num_timesteps=False, tb_log_name=full_id)
        if args["log"]:
            model.save(f"{models_dir}/{args['steps_per_iter']*i}")
            with open('recent_model.txt', 'w') as file:
                file.write(f"{models_dir}/{args['steps_per_iter']*i}")
    """"
    Цикл обучает модель num_iters раз.
    Каждая итерация — steps_per_iter шагов.
    После каждой итерации модель сохраняется в models/{full_id}/{steps_per_iter*i}.
    Последняя модель записывается в recent_model.txt.
    """

    #Закрытие среды. После обучения среда сразу закрывается 
    train_env.close()

if __name__ == "__main__":
    #Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description='Train an agent in an environment using a DRL algorithm from stable baselines 3')

    parser.add_argument('--log', type=bool, default=True, help='record logs to tensorboard')
    parser.add_argument('--algorithm', '-a', type=str, default="PPO", help='which DRL algorithm to use for training')
    parser.add_argument('--environment', '-env', type=str, default="QuadX-Waypoints-v1", help='which environment to train on')
    parser.add_argument('--steps_per_iter', '-spi', type=int, default=10000, help='the number of timesteps for each saved model')
    parser.add_argument('--num_iters', '-ni', type=int, default=100, help='the number of iterations')
    args = parser.parse_args()

    """
    Аргумент                         Описание	                                         Значение по умолчанию
    --log	                         Логировать в TensorBoard?	                         True
    --algorithm (-a)	             Алгоритм DRL (PPO, A2C, DDPG, TD3, SAC)	         "PPO"
    --environment (-env)	         Среда Gymnasium (PyFlyt/QuadX-Waypoints-v4)	     "QuadX-Waypoints-v1"
    --steps_per_iter (-spi)	         Шагов на одну итерацию	10000                        10000
    --num_iters (-ni)	             Число итераций	                                     100
    """

    train(vars(args))

    """
    Как запустить?
        Пример 1: Обучить PPO на QuadX-Waypoints-v4
            python train.py --algorithm PPO --environment QuadX-Waypoints-v4 --steps_per_iter 10000 --num_iters 50

        Пример 2: Обучить SAC без логирования
            python train.py --algorithm SAC --log False
    """

    """"
    Вывод
        Этот скрипт:
            Создаёт среду для обучения квадрокоптера (PyFlyt/QuadX-Waypoints-v4).
            Инициализирует выбранный алгоритм DRL (например, PPO).
            Обучает модель и сохраняет чекпоинты.
            Логирует процесс в TensorBoard (если --log=True).
            
        Если есть ошибки (например, Environment not found), проверьте:
            Правильно ли указано имя среды (QuadX-Waypoints-v4 вместо recent_model.txt).
            Установлены ли PyFlyt и gymnasium.
    """