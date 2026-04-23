import pandas as pd
import matplotlib.pyplot as plt
import pdb
from config import EnvConfig
from envs import make_env
from Basilisk.architecture import bskLogging

def main():
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    cfg = EnvConfig()
    env = make_env(cfg)

    

    obs, info = env.reset(seed = cfg.seed)
    print("Initial Observation:", obs)
    print("Initial info:", info)

    #pdb.set_trace()

    total_reward = 0.0
    step_count = 0

    print(env.action_space)
    
    # Dumb hand-coded policy:
    # alternate between charge and scan
    # assuming Discrete action indexing in declared order
    # action 0 = Charge, action 1 = Scan
    done = False
    records = [];
    while not done and step_count < 100:
        action = env.action_space.sample()
        print("Random Action", action)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1
        done = terminated or truncated

        
        records.append({
            "step": step_count,
            "action": action,
            "charge": obs[0],
            "reward": reward,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
        })

    print(f"\nFinished after {step_count} steps")
    print(f"Total reward: {total_reward:.4f}")

    env.close()

    df = pd.DataFrame(records)
    print(df)

    fig, axes = plt.subplots(3, 1, figsize=(10,8), sharex=True)
    axes[0].plot(df["step"], df["action"])
    axes[0].set_ylabel("Action")

    axes[1].plot(df["step"], df["reward"], label='Reward')
    axes[1].plot(df["step"], df["total_reward"], label='Cumulative Reward')
    axes[1].set_ylabel("Reward")

    axes[2].plot(df["step"], df["charge"])
    axes[2].set_ylabel("Charge")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()