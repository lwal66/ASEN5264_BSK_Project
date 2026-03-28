import pandas as pd
import matplotlib.pyplot as plt

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

    total_reward = 0.0
    step_count = 0


    
    # Dumb hand-coded policy:
    # alternate between charge and scan
    # assuming Discrete action indexing in declared order
    # action 0 = Charge, action 1 = Scan
    done = False
    records = [];
    while not done and step_count < 20:
        action = int(step_count % 2)
        # if step_count == 5 or step_count == 10 :
        #     action = 0
        # else:
        #     action = 1


        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1
        done = terminated or truncated

        records.append({
            "step": step_count,
            "action": action,
            "reward": reward,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "battery": obs[0],
            "storage": obs[1],
            "eclipse_start": obs[2],
            "eclipse_end": obs[3]
        })

    print(f"\nFinished after {step_count} steps")
    print(f"Total reward: {total_reward:.4f}")

    env.close()

    df = pd.DataFrame(records)
    print(df)

    fig, axes = plt.subplots(5, 1, figsize=(10,8), sharex=True)
    axes[0].plot(df["step"], df["action"])
    axes[0].set_ylabel("Action")

    axes[1].plot(df["step"], df["battery"])
    axes[1].set_ylabel("Battery")

    axes[2].plot(df["step"], df["storage"])
    axes[2].set_ylabel("Storage")

    axes[3].plot(df["step"], df["reward"], label='Reward')
    axes[3].plot(df["step"], df["total_reward"], label='Cumulative Reward')
    axes[3].set_ylabel("Reward")

    axes[4].plot(df["step"], df["eclipse_start"], label='eclipse_start')
    axes[4].plot(df["step"], df["eclipse_end"], label='eclipse_end')
    axes[4].set_ylabel("Eclipse start and end")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()