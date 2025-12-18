Этот проект - реализация асинхронной версии алгоритма PPO. Чтобы запустить нужно:

1. удостовериться, что все requirements из requirements.txt установлены (запускать в conda или venv)

2. рекомендуется установить следующие вещи:

```
!pip install "gymnasium[atari, accept-rom-license]"
!pip install "gymnasium[other]"
```

3. Запустить один из скриптов experiments_sync.py или experiments_async.py. В них можно указать параметры обучения вручную, например:

```
if __name__ == "__main__":

    frequency = 1 #update frequency
    maxsize = 5 #queue_maxsize
```
4. Логи будут сохраняться в соответствующий csv файл. В ноутбуке report.ipynb содержится отчёт об экспериментах
