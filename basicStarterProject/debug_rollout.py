from config import EnvConfig
from envs import make_env

def main():
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
    while not done and step_count < 20:
        action = int(step_count % 2)
        print(f"action={action} ")
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1
        done = terminated or truncated

        print(
            f"step={step_count} "
            f"action={action} "
            f"reward={reward:.4f} "
            f"terminated={terminated} "
            f"truncated={truncated}"
        )
        print("obs:", obs)

    print(f"\nFinished after {step_count} steps")
    print(f"Total reward: {total_reward:.4f}")

    env.close()


if __name__ == "__main__":
    main()