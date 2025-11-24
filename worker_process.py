import time
import queue
import torch
import gymnasium as gym
import ale_py
from torch import multiprocessing as mp

from learner import PPOLearning
from model import PPOActorCritic

def put_overwrite(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

def worker_process(worker_id, rollout_queue, shared_weights_dict, stop_event, n_states, device, maxsize):
    torch.set_num_threads(1)
    
    gym.register_envs(ale_py)
    env = gym.make("PongNoFrameskip-v4")
    model = PPOActorCritic(n_states).to(device)
    learner = PPOLearning(model, env, device)

    print(f"[Worker {worker_id}] Started.")
    local_version = -1

    while not stop_event.is_set():
        try:
            current_version = shared_weights_dict.get('version', -1)
            
            if current_version > local_version:
                state_dict = shared_weights_dict.get('weights')
                if state_dict is not None:
                    learner.actorcritic.load_state_dict(state_dict)
                    learner.version = current_version
                    local_version = current_version
        except Exception as e:
            pass

        try:
            rollout = learner.create_rollout(length=2048)
        except Exception as e:
            print(f"[Worker {worker_id}] Error in rollout: {e}")
            break

        item = (rollout, learner.version)
        put_overwrite(rollout_queue, item)
        time.sleep(0.01)

    env.close()
    print(f"[Worker {worker_id}] Finished.")
