import gymnasium  # Библиотека для создания сред RL
import os  # Для работы с файловой системой
import argparse  # Для обработки аргументов командной строки

from PyFlyt.gym_envs import FlattenWaypointEnv  # Среда для управления квадрокоптером
from stable_baselines3.common.env_util import make_vec_env  # Создание векторной среды
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC  # Алгоритмы DRL

def eval(args):
    
    environment = args['environment']

    print(args['model'])

    #Определение алгоритма из пути к модели
    algorithm_name = args['model'][args['model'].find('/')+1:args['model'].find('_')]
    if algorithm_name == "PPO": algorithm = PPO
    elif algorithm_name == "A2C": algorithm = A2C
    elif algorithm_name == "DDPG": algorithm = DDPG
    elif algorithm_name == "TD3": algorithm = TD3
    elif algorithm_name == "SAC": algorithm = SAC
    else:
        print("Error: Invalid DRL Algorithm specified")
        return
    """"
    Из пути к модели (например, models/PPO_QuadX-Waypoints-v1_abc123/100000) извлекается название алгоритма (PPO, A2C и т.д.).
    На его основе выбирается соответствующий класс алгоритма из Stable Baselines3.
    """
    #Загрузка модели
    model_path = f"{args['model']}"

    env = make_vec_env(lambda: FlattenWaypointEnv(gymnasium.make(f"PyFlyt/{environment}"), context_length=1), n_envs=1)
    model = algorithm.load(model_path, env=env)
    """"
    Создаётся среда PyFlyt/{environment} (например, PyFlyt/QuadX-Waypoints-v1).
    Модель загружается из указанного пути (args['model']).
    """

    #Запуск эпизодов оценки
    for _ in range(args['eval_episodes']):
        render_env = FlattenWaypointEnv(gymnasium.make(f"PyFlyt/{environment}", render_mode="human"), context_length=1)
        obs = render_env.reset()
        obs = obs[0]
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = render_env.step(action)
            done = terminated or truncated
            render_env.render()
        render_env.close()

        """
        Запускается eval_episodes эпизодов (по умолчанию 10).
        Для каждого эпизода:
            Создаётся новая среда с визуализацией (render_mode="human").
            Модель предсказывает действия на основе наблюдения (obs).
            Шаг симуляции выполняется через env.step(action).
            Цикл продолжается, пока эпизод не завершится (terminated или truncated).
            Среда закрывается после эпизода.
        """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained agent in an environment')

    with open('recent_model.txt', 'r') as file:
        recent_model = file.read().strip()

    parser.add_argument('--model', '-m', type=str, default=recent_model, help='which saved model to evaluate')
    parser.add_argument('--environment', '-env', type=str, default="QuadX-Waypoints-v1", help='which environment to train on')
    parser.add_argument('--eval_episodes', '-ee', type=int, default=10, help='the number of episodes to evaluate on')
    args = parser.parse_args()

    eval(vars(args))

    """
    Доступные аргументы:

    Аргумент	                Описание	                                    Значение по умолчанию
    --model (-m)	            Путь к сохранённой модели	                    Берётся из recent_model.txt
    --environment (-env)	    Среда Gymnasium (PyFlyt/QuadX-Waypoints-v1)	    "QuadX-Waypoints-v1"
    --eval_episodes (-ee)	    Количество эпизодов для оценки	                10  
    """


    """
    4. Как запустить?
        Пример 1: Оценка последней обученной модели
            python eval.py
            (модель берётся из recent_model.txt)

        Пример 2: Оценка конкретной модели  
            python eval.py --model models/PPO_QuadX-Waypoints-v1_abc123/100000 --environment QuadX-Waypoints-v4 --eval_episodes 5
    
     
    Вывод
        Этот скрипт:
            Загружает предобученную модель (PPO, SAC и др.).
            Создаёт среду PyFlyt с визуализацией.
            Запускает указанное число эпизодов, показывая, как модель управляет квадрокоптером.
            
        Если есть ошибки (например, Environment not found), проверьте:
            Правильно ли указано имя среды (QuadX-Waypoints-v4 вместо recent_model.txt).
            Существует ли загружаемая модель.
    """