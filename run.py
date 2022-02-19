import argparse
import datetime
import os
import time
from collections import deque

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch
import numpy as np

from env.utils import make_env, make_vec_envs
from com.logger import CSVLogger

now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
device = "cuda" if torch.cuda.is_available() else "cpu"


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "play", "test"])
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--env", type=str, default="env:IIWACustomEnv-v0")
    parser.add_argument("--dir", type=str, default=os.path.join("exp", now_str))
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--frames", type=int, default=2.5e7)
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--plot", type=int, default=0)
    parser.add_argument("--use_curriculum", type=int, default=1)
    return parser


def run_ppo(args):
    from algo.ppo import PPO, PPOReplayBuffer, SoftsignActor, Policy

    num_frames = args.frames
    episode_steps = 40000
    num_processes = 125 if os.name != "nt" else torch.multiprocessing.cpu_count()
    num_steps = episode_steps // num_processes
    mini_batch_size = 2000
    num_mini_batch = episode_steps // mini_batch_size
    save_every = int(num_frames // 5)
    save_name = "-".join(args.env.split(":"))

    ppo_params = {
        "use_clipped_value_loss": False,
        "num_mini_batch": num_mini_batch,
        "entropy_coef": 0.0,
        "value_loss_coef": 1.0,
        "ppo_epoch": 10,
        "clip_param": 0.2,
        "lr": 3e-4,
        "eps": 1e-5,
        "max_grad_norm": 2.0,
    }

    envs = make_vec_envs(args.env, 1234, num_processes)

    obs_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0]

    dummy_env = make_env(args.env)

    if args.net is None:
        policy = Policy(SoftsignActor(dummy_env)).to(device)
    else:
        policy = torch.load(args.net, map_location=device)

    agent = PPO(policy, **ppo_params)

    mirror_indices = dummy_env.unwrapped.get_mirror_indices()
    rollouts = PPOReplayBuffer(
        num_steps,
        num_processes,
        obs_dim,
        action_dim,
        device=device,
        mirror=mirror_indices,
    )

    ep_rewards = deque(maxlen=num_processes)
    curriculum_metrics = deque(maxlen=num_processes)
    num_updates = int(num_frames) // num_steps // num_processes

    # don't divide by 0
    ep_rewards.append(0)
    curriculum_metrics.append(0)

    # This has to be done before reset
    if args.use_curriculum:
        current_curriculum = dummy_env.unwrapped.curriculum
        max_curriculum = dummy_env.unwrapped.max_curriculum
        advance_threshold = dummy_env.unwrapped.advance_threshold
        envs.set_env_params({"curriculum": current_curriculum})
        del dummy_env
    else:
        current_curriculum = -1
        max_curriculum = -1
        advance_threshold = -1

    obs = envs.reset()
    rollouts.observations[0].copy_(torch.from_numpy(obs))

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    logger = CSVLogger(log_dir=args.dir)
    save_checkpoint = save_every
    best_reward_so_far = float("-inf")
    start_time = time.time()

    for iteration in range(num_updates):

        scheduled_lr = max(ppo_params["lr"] * (0.99 ** iteration), 3e-5)
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = scheduled_lr

        # Disable gradient for data collection
        torch.set_grad_enabled(False)
        policy.train(mode=False)

        for (
            env_obs,
            obs_buf,
            act_buf,
            act_log_prob_buf,
            value_buf,
            reward_buf,
            mask_buf,
            bad_mask_buf,
        ) in zip(
            rollouts.observations[:-1],
            rollouts.observations[1:],
            rollouts.actions,
            rollouts.action_log_probs,
            rollouts.value_preds,
            rollouts.rewards,
            rollouts.masks[1:],
            rollouts.bad_masks[1:],
        ):
            value, action, action_log_prob = policy.act(env_obs)
            cpu_actions = action.cpu().numpy()

            obs, reward, done, info = envs.step(cpu_actions)

            mask = torch.from_numpy((~done).astype(np.float32)[:, None])
            reward = torch.from_numpy(reward.astype(np.float32)[:, None])
            bad_mask = torch.tensor(
                [0.0 if "bad_transition" in d else 1.0 for d in info]
            ).view(-1, 1)
            ep_rewards.extend([d["episode"]["r"] for d in info if "episode" in d])
            curriculum_metrics.extend(
                [d["curriculum_metric"] for d in info if "curriculum_metric" in d]
            )

            obs_buf.copy_(torch.from_numpy(obs))
            act_buf.copy_(action)
            act_log_prob_buf.copy_(action_log_prob)
            value_buf.copy_(value)
            reward_buf.copy_(reward)
            mask_buf.copy_(mask)
            bad_mask_buf.copy_(bad_mask)

        next_value = policy.get_value(rollouts.observations[-1]).detach()

        # Update curriculum after roll-out
        mean_curriculum_metric = sum(curriculum_metrics) / len(curriculum_metrics)
        if (
            args.use_curriculum
            and mean_curriculum_metric > advance_threshold
            and current_curriculum < max_curriculum
        ):
            current_curriculum += 1
            envs.set_env_params({"curriculum": current_curriculum})
            curriculum_metrics.clear()
            curriculum_metrics.append(0)  # append 0 to make sure we don't divide by 0

        # Enable gradients for training
        torch.set_grad_enabled(True)
        policy.train(mode=True)

        rollouts.compute_returns(next_value)
        _, _, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        model_name = f"{save_name}_latest.pt"
        torch.save(policy, os.path.join(args.dir, model_name))

        frame_count = (iteration + 1) * num_steps * num_processes
        if frame_count >= save_checkpoint or iteration == num_updates - 1:
            model_name = f"{save_name}_{int(save_checkpoint)}.pt"
            save_checkpoint += save_every
            torch.save(policy, os.path.join(args.dir, model_name))

        mean_ep_reward = sum(ep_rewards) / len(ep_rewards)
        if len(ep_rewards) > 1 and mean_ep_reward > best_reward_so_far:
            best_reward_so_far = mean_ep_reward
            model_name = f"{save_name}_best.pt"
            torch.save(policy, os.path.join(args.dir, model_name))

        if len(ep_rewards) > 1:
            elapsed_time = time.time() - start_time
            fps = int(frame_count / elapsed_time)
            print(
                f"Steps: {frame_count:d} | FPS: {fps:d} |",
                f"Mean: {mean_ep_reward:.1f} | Max: {max(ep_rewards):.1f} |",
                f"Cur: {current_curriculum:2d} | CurM: {mean_curriculum_metric:.1f}",
                flush=True,
            )
            logger.log_epoch(
                {
                    "iter": iteration + 1,
                    "total_num_steps": frame_count,
                    "fps": fps,
                    "entropy": dist_entropy,
                    "curriculum": current_curriculum,
                    "curriculum_metric": mean_curriculum_metric,
                    "stats": {"rew": ep_rewards},
                }
            )

    envs.close()


def play(args):
    policy = torch.load(args.net, map_location="cpu")
    controller = policy.actor

    render = args.render == 1
    env = make_env(args.env, render=render)
    env.unwrapped.curriculum = 0

    obs = env.reset()

    # Set global no_grad
    torch.set_grad_enabled(False)
    policy.train(mode=False)

    ep_reward = 0
    while True:
        if not render or not env.camera.env_should_wait:
            obs = torch.from_numpy(obs).float().unsqueeze(0)

            action = controller(obs)
            cpu_actions = action.squeeze().cpu().numpy()

            obs, reward, done, _ = env.step(cpu_actions)
            ep_reward += reward

            if done:
                print("--- Episode reward:", ep_reward)
                obs = env.reset()
                ep_reward = 0

        if render:
            env.camera.wait()
            env.unwrapped._handle_keyboard()


def test(args):
    render = args.render == 1
    env = make_env(args.env, render=render)

    bc = env.unwrapped._p

    env.reset()
    while True:
        if not render or not env.camera.env_should_wait:
            action = env.action_space.sample() * 1
            obs, rew, done, info = env.step(action)

            if done:
                env.reset()

        if render:
            env.camera.wait()
            env.unwrapped._handle_keyboard()


def train(args):
    run_ppo(args)


if __name__ == "__main__":
    args = arg_parser().parse_args()
    (globals().get(args.mode))(args)
