import datetime

import torch, os
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from sam_env.logger import Logger
from trainer import algo, utils
import time
import numpy as np
from evaluation import evaluate


def train(training_data, args, max_reward=-np.inf):

    agent, actor_critic, rollouts, envs = training_data
    device = torch.device("cuda:0" if args.cuda else "cpu")

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"

    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    e = datetime.datetime.now()
    time_formatted = e.strftime("_%Y-%m-%d_%H:%M:%S")

    logger = Logger(os.path.join('logs', args.algo + "_" + args.env_name + time_formatted))

    episode_rewards = deque(maxlen=100)
    episode_length = deque(maxlen=100)

    obs = envs.reset()

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    total_iterations = int(args.num_env_steps) // args.num_steps // args.num_processes

    start = time.time()

    best_reward = -np.inf
    for j in range(total_iterations):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(agent.optimizer, j, total_iterations, args.lr)
            logger.write('lr', agent.optimizer.param_groups[0]['lr'])

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            logger.step()

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_length.append(info['episode']['l'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        end = time.time()

        rollouts.after_update()

        if np.mean(episode_rewards) >= max_reward:
            print('early stopped, hit max reward')
            break

        # save for every interval-th episode or for the last epoch
        if len(episode_rewards) > 1 and args.save_dir != "":
            rew = np.mean(episode_rewards)
            if rew > best_reward:
                best_reward = rew

                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: "
                "mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                "mean/median length {:.1f}/{:.1f}, min/max length {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards),
                            np.mean(episode_length),
                            np.median(episode_length), np.min(episode_length),
                            np.max(episode_length), dist_entropy, value_loss,
                            action_loss))
            logger.write('loss/value', value_loss)
            logger.write('loss/policy', action_loss)
            logger.write('experiment/num_updates', j)
            logger.write('experiment/FPS', int(total_num_steps / (end - start)))
            logger.write('experiment/EPISODE MEAN', np.mean(episode_rewards))
            logger.write('experiment/EPISODE MEDIAN', np.median(episode_rewards))
            logger.write('experiment/EPISODE MIN', np.min(episode_rewards))
            logger.write('experiment/EPSIDOE MAX', np.max(episode_rewards))

        if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

        logger.close()