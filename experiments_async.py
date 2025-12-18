import torch
import gymnasium as gym
import numpy as np
from torch import nn
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import ale_py
import csv

import os
from rollout_buffer import RolloutBuffer
from python_dataset import PythonListDataset
from model import PPOActorCritic
from learner import PPOLearning
import time
from tqdm import tqdm
import torch.multiprocessing as mp
from learner_process import learner_process
from worker_process import worker_process

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

gym.register_envs(ale_py)

import time
import queue
import pickle
import torch
import torch.multiprocessing as mp

from worker_process import worker_process
from learner_process import learner_process

def test(frequency=1, maxsize=16, duration_seconds=None, max_iterations=50):
    torch.set_num_threads(1)
    
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu") 

    manager = mp.Manager()
    shared_weights_dict = manager.dict()
    stop_event = mp.Event()

    rollout_queue = mp.Queue(maxsize=maxsize)
    results_queue = mp.Queue()

    num_workers = 4 
    workers = []

    print(f"Launching {num_workers} workers...")
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(wid, rollout_queue, shared_weights_dict, stop_event, 6, cpu_device, maxsize)
        )
        p.start()
        workers.append(p)

    print("Launching learner...")
    learner = mp.Process(
        target=learner_process,
        args=(rollout_queue, shared_weights_dict, stop_event, 6, frequency, device, results_queue, max_iterations)
    )
    learner.start()

    collected_results = []
    learner_finished = False
    start_time = time.time()

    pkl_filename = f"results_async_fr{frequency}_ms{maxsize}.pkl"
    csv_filename = f"results_async_fr{frequency}_ms{maxsize}.csv"

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'iterations', 'updates', 'reward', 'queue_size', 'rollout_version'])
    
    print(f"Collecting results... (Real-time logs in {csv_filename})")

    while not learner_finished:
        try:
            msg = results_queue.get(timeout=1.0)
            
            if msg == "DONE":
                learner_finished = True
                print("Main: Received DONE signal.")
            else:
                collected_results.append(msg)

                try:
                    with open(csv_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(msg)
                except Exception as e:
                    print(f"CSV Error: {e}")

                if len(collected_results) % 10 == 0:
                    try:
                        temp_pkl = pkl_filename + ".tmp"
                        with open(temp_pkl, 'wb') as file:
                            pickle.dump(collected_results, file)
                        os.replace(temp_pkl, pkl_filename)
                    except Exception as e:
                        print(f"PKL Auto-save Error: {e}")
                
        except queue.Empty:
            if not learner.is_alive() and results_queue.empty():
                print("Main: Learner died unexpectedly!")
                learner_finished = True
            
            if duration_seconds and (time.time() - start_time > duration_seconds):
                print("Main: Time limit reached.")
                stop_event.set()
                learner_finished = True

    print("Terminating processes...")
    stop_event.set()
 
    time.sleep(1)
    
    if learner.is_alive(): learner.terminate()
    learner.join()

    for w in workers:
        if w.is_alive(): w.terminate()
        w.join()

    try:
        with open(pkl_filename, 'wb') as file:
            pickle.dump(collected_results, file)
        print(f"Final Save: {len(collected_results)} entries to {pkl_filename}")
    except Exception as e:
        print(f"Error saving file: {e}")

    return collected_results

import pandas as pd
import numpy as np

import os
import pickle

def save_array_to_csv(array, x, y):
    filename = f'results_{x}_{y}.csv'
    df = pd.DataFrame(array, columns=['time', 'iterations', 'updates', 'reward', 'queue_size', 'rollout_version'])
    
    df.to_csv(filename, index=False)
    print(f"Данные сохранены в файл: {filename}")
    print(f"Размер данных: {array.shape}")
    return df

if __name__ == "__main__":

    frequency = 1
    maxsize = 5

    results = test(frequency, maxsize, duration_seconds = 3600 * 6, max_iterations=30000)

    def load_pkl(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    results = load_pkl(f'/home/tilin/Theorcos/results_async_fr{frequency}_ms{maxsize}.pkl')
    print(results)
    save_array_to_csv(np.array(results), frequency , maxsize)
