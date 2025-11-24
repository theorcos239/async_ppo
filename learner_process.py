import time
import queue
import torch
import gymnasium as gym
import ale_py
from torch import multiprocessing as mp

from learner import PPOLearning
from model import PPOActorCritic

def get_weights(model):
    return {k: v.cpu() for k, v in model.state_dict().items()}

def learner_process(rollout_queue, shared_weights_dict, stop_event, n_actions, frequency, device, results_queue, max_iterations=1000):
    torch.set_num_threads(4)

    gym.register_envs(ale_py)
    env = gym.make("PongNoFrameskip-v4")
    model = PPOActorCritic(n_actions).to(device)
    learner = PPOLearning(model, env, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print("[Learner] Sending initial weights...")
    shared_weights_dict['weights'] = get_weights(model)
    shared_weights_dict['version'] = 0

    print("[Learner] Started training loop.")
    tic = time.time()
    iteration = 0

    try:
        while iteration < max_iterations:
            try:
                r, version = rollout_queue.get(timeout=10.0)
            except queue.Empty:
                if stop_event.is_set(): break
                continue

            learner.one_epoch_of_learning(r, optimizer)

            total_reward = learner.testing()
            tac = time.time()
            iteration += 1

            log_entry = [
                tac - tic, 
                iteration * len(r), 
                iteration // frequency, 
                total_reward, 
                rollout_queue.qsize(),
                version
            ]
            results_queue.put(log_entry)
            
            print(f"[Learner] Iteration {iteration}/{max_iterations} | Reward: {total_reward:.2f}")

            if iteration % frequency == 0:
                shared_weights_dict['weights'] = get_weights(model)
                shared_weights_dict['version'] = iteration

    except Exception as e:
        print(f"[Learner] Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        stop_event.set()
        results_queue.put("DONE")
        print("[Learner] Finished.")
