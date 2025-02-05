"""
HIRO training process
"""
import os
import datetime
from torch import Tensor
import copy
import torch
from torch.nn import functional
import numpy as np
import wandb
from utils import get_env, get_target_position, log_video_hrl, ParamDict, LoggerTrigger, TimeLogger, print_cmd_hint
from network import ActorLow, ActorHigh, CriticLow, CriticHigh
from experience_buffer import ExperienceBufferLow, ExperienceBufferHigh
import pdb
from torch import nn

class DilatedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, radius=10, device='cuda'):
        super().__init__()
        self.radius = radius
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size).to(device)
        self.index = torch.arange(0, radius * hidden_size, radius)
        self.dilation = 0
        self.device = device

    def forward(self, state, hidden):
        """At each time step only the corresponding part of the state is updated
        and the output is pooled across the previous c out- puts."""
        d_idx = self.dilation_idx.to(self.device)
        hx, cx = hidden
        hx[:, d_idx], cx[:, d_idx] = self.rnn(state, (hx[:, d_idx], cx[:, d_idx]))
        detached_hx = hx[:, self.masked_idx(d_idx)].detach()
        detached_hx = detached_hx.view(detached_hx.shape[0], self.hidden_size, self.radius-1)
        detached_hx = detached_hx.sum(-1)

        y = (hx[:, d_idx] + detached_hx) / self.radius
        return y, (hx, cx)

    def masked_idx(self, dilated_idx):
        """Because we do not want to have gradients flowing through all
        parameters but only at the dilation index, this function creates a
        'negated' version of dilated_index, everything EXCEPT these indices."""
        masked_idx = torch.arange(1, self.radius * self.hidden_size + 1)
        masked_idx[dilated_idx] = 0
        masked_idx = masked_idx.nonzero()
        masked_idx = masked_idx - 1
        return masked_idx

    @property
    def dilation_idx(self):
        """Keep track at which dilation we currently we are."""
        dilation_idx = self.dilation + self.index
        self.dilation = (self.dilation + 1) % self.radius
        return dilation_idx


class Policy_Network(nn.Module):
    def __init__(self, d, time_horizon, num_workers, device):
        super().__init__()
        self.device = device
        self.Mrnn = DilatedLSTM(13, 1, time_horizon, device=device).to(device)
        self.num_workers = num_workers

    def forward(self, z, goal_5_norm, goal_4_norm, goal_3_norm, hidden, mask):
        goal_x_info = torch.cat(([goal_5_norm.unsqueeze(1).detach(), goal_4_norm.unsqueeze(1).detach(), goal_3_norm.unsqueeze(1).detach(), z.reshape(3,10)]), dim=1).to(self.device)
        hidden = (mask * hidden[0], mask * hidden[1])
        policy_network_result, hidden = self.Mrnn(goal_x_info, hidden)
        #pdb.set_trace()
        policy_network_result = (policy_network_result - policy_network_result.detach().min()) / \
                                (policy_network_result.detach().max()-
                                 policy_network_result.detach().min())
        #policy_network_result = torch.nn.functional.softmax(policy_network_result, dim=0)
        #policy_network_result[policy_network_result < 0.5] = 0
        #policy_network_result[policy_network_result > 0.5] = 1
        policy_network_result = policy_network_result.round()

        return policy_network_result.type(torch.float), hidden


def init_hidden(n_workers, h_dim, device, grad=False):
    return (torch.zeros(n_workers, h_dim, requires_grad=grad).to(device),
            torch.zeros(n_workers, h_dim, requires_grad=grad).to(device))

def save_evaluate_utils(step, actor_l, actor_h, params, file_path=None, file_name=None):
    if file_name is None:
        time = datetime.datetime.now()
        file_name = "evalutils-hiro-{}_{}-it({})-[{}].tar".format(params.env_name.lower(), params.prefix, step, time)
    if file_path is None:
        file_path = os.path.join("rl_hiro", "save", "model", file_name)
    print("\n    > saving evaluation utils...")
    torch.save({
        'step': step,
        'actor_l': actor_l.state_dict(),
        'actor_h': actor_h.state_dict(),
    }, file_path)
    print("    > saved evaluation utils to: {}\n".format(file_path))


def save_checkpoint(step, actor_l, critic_l, actor_optimizer_l, critic_optimizer_l, exp_l, actor_h, critic_h, actor_optimizer_h, critic_optimizer_h, exp_h, logger, params, file_path=None, file_name=None):
    if file_name is None:
        time = datetime.datetime.now()
        file_name = "checkpoint-hiro-{}_{}-it({})-[{}].tar".format(params.env_name.lower(), params.prefix, step, time)
    if file_path is None:
        file_path = os.path.join("rl_hiro", "save", "model", file_name)
    print("\n    > saving training checkpoint...")
    torch.save({
        'step': step,
        'params': params,
        'logger': logger,
        'actor_l': actor_l.state_dict(),
        'critic_l': critic_l.state_dict(),
        'actor_optimizer_l': actor_optimizer_l.state_dict(),
        'critic_optimizer_l': critic_optimizer_l.state_dict(),
        'exp_l': exp_l,
        'actor_h': actor_h.state_dict(),
        'critic_h': critic_h.state_dict(),
        'actor_optimizer_h': actor_optimizer_h.state_dict(),
        'critic_optimizer_h': critic_optimizer_h.state_dict(),
        'exp_h': exp_h
    }, file_path)
    print("    > saved checkpoint to: {}\n".format(file_path))


def load_checkpoint(file_name):
    try:
        # load checkpoint file
        print("\n    > loading training checkpoint...")
        file_path = os.path.join("rl_hiro", "save", "model", file_name)
        checkpoint = torch.load(file_path)
        print("\n    > checkpoint file loaded! parsing data...")
        params = checkpoint['params']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
        # load utils
        policy_params = params.policy_params
        state_dim = params.state_dim
        goal_dim = params.goal_dim
        action_dim = params.action_dim
        max_action = policy_params.max_action
        max_goal = policy_params.max_goal
        # initialize rl components
        actor_eval_l = ActorLow(state_dim, goal_dim, action_dim, max_action).to(device)
        actor_optimizer_l = torch.optim.Adam(actor_eval_l.parameters(), lr=policy_params.actor_lr)
        critic_eval_l = CriticLow(state_dim, goal_dim, action_dim).to(device)
        critic_optimizer_l = torch.optim.Adam(critic_eval_l.parameters(), lr=policy_params.critic_lr)

        actor_eval_2 = ActorHigh(state_dim, goal_dim, max_goal, device).to(device)
        actor_optimizer_2 = torch.optim.Adam(actor_eval_2.parameters(), lr=policy_params.actor_lr)
        critic_eval_2 = CriticHigh(state_dim, goal_dim).to(device)
        critic_optimizer_2 = torch.optim.Adam(critic_eval_2.parameters(), lr=policy_params.critic_lr)

        actor_eval_3 = ActorHigh(state_dim, goal_dim, max_goal, device).to(device)
        actor_optimizer_3 = torch.optim.Adam(actor_eval_3.parameters(), lr=policy_params.actor_lr)
        critic_eval_3 = CriticHigh(state_dim, goal_dim).to(device)
        critic_optimizer_3 = torch.optim.Adam(critic_eval_3.parameters(), lr=policy_params.critic_lr)

        actor_eval_4 = ActorHigh(state_dim, goal_dim, max_goal, device).to(device)
        actor_optimizer_4 = torch.optim.Adam(actor_eval_4.parameters(), lr=policy_params.actor_lr)
        critic_eval_4 = CriticHigh(state_dim, goal_dim).to(device)
        critic_optimizer_4 = torch.optim.Adam(critic_eval_4.parameters(), lr=policy_params.critic_lr)

        actor_eval_5 = ActorHigh(state_dim, goal_dim, max_goal, device).to(device)
        actor_optimizer_5 = torch.optim.Adam(actor_eval_5.parameters(), lr=policy_params.actor_lr)
        critic_eval_5 = CriticHigh(state_dim, goal_dim).to(device)
        critic_optimizer_5 = torch.optim.Adam(critic_eval_5.parameters(), lr=policy_params.critic_lr)
        # unpack checkpoint object
        step = checkpoint['step'] + 1
        logger = checkpoint['logger']
        #
        actor_eval_l.load_state_dict(checkpoint['actor_l'])
        critic_eval_l.load_state_dict(checkpoint['critic_l'])
        actor_optimizer_l.load_state_dict((checkpoint['actor_optimizer_l']))
        critic_optimizer_l.load_state_dict(checkpoint['critic_optimizer_l'])
        experience_buffer_l = checkpoint['exp_l']
        #
        actor_eval_2.load_state_dict(checkpoint['actor_2'])
        critic_eval_2.load_state_dict(checkpoint['critic_2'])
        actor_optimizer_2.load_state_dict((checkpoint['actor_optimizer_2']))
        critic_optimizer_2.load_state_dict(checkpoint['critic_optimizer_2'])

        actor_eval_3.load_state_dict(checkpoint['actor_3'])
        critic_eval_3.load_state_dict(checkpoint['critic_3'])
        actor_optimizer_3.load_state_dict((checkpoint['actor_optimizer_3']))
        critic_optimizer_3.load_state_dict(checkpoint['critic_optimizer_3'])

        actor_eval_4.load_state_dict(checkpoint['actor_4'])
        critic_eval_4.load_state_dict(checkpoint['critic_4'])
        actor_optimizer_4.load_state_dict((checkpoint['actor_optimizer_4']))
        critic_optimizer_4.load_state_dict(checkpoint['critic_optimizer_4'])

        actor_eval_5.load_state_dict(checkpoint['actor_5'])
        critic_eval_5.load_state_dict(checkpoint['critic_5'])
        actor_optimizer_5.load_state_dict((checkpoint['actor_optimizer_5']))
        critic_optimizer_5.load_state_dict(checkpoint['critic_optimizer_5'])

        experience_buffer_2 = checkpoint['exp_2']
        experience_buffer_3 = checkpoint['exp_3']
        experience_buffer_4 = checkpoint['exp_4']
        experience_buffer_5 = checkpoint['exp_5']
        #
        actor_target_l = copy.deepcopy(actor_eval_l).to(device)
        critic_target_l = copy.deepcopy(critic_eval_l).to(device)
        actor_target_2 = copy.deepcopy(actor_eval_2).to(device)
        critic_target_2 = copy.deepcopy(critic_eval_2).to(device)

        actor_target_3 = copy.deepcopy(actor_eval_3).to(device)
        critic_target_3 = copy.deepcopy(critic_eval_3).to(device)

        actor_target_4 = copy.deepcopy(actor_eval_4).to(device)
        critic_target_4 = copy.deepcopy(critic_eval_4).to(device)

        actor_target_5 = copy.deepcopy(actor_eval_5).to(device)
        critic_target_5 = copy.deepcopy(critic_eval_5).to(device)
        #
        actor_eval_l.train(), actor_target_l.train(), critic_eval_l.train(), critic_target_l.train()
        actor_eval_2.train(), actor_target_2.train(), critic_eval_2.train(), critic_target_2.train()
        actor_eval_3.train(), actor_target_3.train(), critic_eval_3.train(), critic_target_3.train()
        actor_eval_4.train(), actor_target_4.train(), critic_eval_4.train(), critic_target_4.train()
        actor_eval_5.train(), actor_target_5.train(), critic_eval_5.train(), critic_target_5.train()
        print("    > checkpoint resume success!")
    except Exception as e:
        print(e)

        policy_network = Policy_Network(state_dim, 1, 1, device).load_state_dict(checkpoint['policy_network'])
    return [step, params, device, logger,
            actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,
            actor_eval_2, actor_target_2, actor_optimizer_2, critic_eval_2, critic_target_2, critic_optimizer_2, experience_buffer_2,
            actor_eval_3, actor_target_3, actor_optimizer_3, critic_eval_3, critic_target_3, critic_optimizer_3, experience_buffer_3,
            actor_eval_4, actor_target_4, actor_optimizer_4, critic_eval_4, critic_target_4, critic_optimizer_4, experience_buffer_4,
            actor_eval_5, actor_target_5, actor_optimizer_5, critic_eval_5, critic_target_5, critic_optimizer_5, experience_buffer_5,
            policy_network]


def initialize_params(params, device):
    policy_params = params.policy_params
    env_name = params.env_name
    max_goal = Tensor(policy_params.max_goal).to(device)
    action_dim = params.action_dim
    goal_dim = params.goal_dim
    max_action = policy_params.max_action
    expl_noise_std_l = policy_params.expl_noise_std_l
    expl_noise_std_h = policy_params.expl_noise_std_h
    c = policy_params.c
    episode_len = policy_params.episode_len
    max_timestep = policy_params.max_timestep
    start_timestep = policy_params.start_timestep
    batch_size = policy_params.batch_size
    sample_n = policy_params.sample_n
    log_interval = params.log_interval
    checkpoint_interval = params.checkpoint_interval
    evaluation_interval = params.evaluation_interval
    save_video = params.save_video
    video_interval = params.video_interval
    env = get_env(params.env_name)
    video_log_trigger = LoggerTrigger(start_ind=policy_params.start_timestep)
    state_print_trigger = LoggerTrigger(start_ind=policy_params.start_timestep)
    checkpoint_logger = LoggerTrigger(start_ind=policy_params.start_timestep, first_log=False)
    evalutil_logger = LoggerTrigger(start_ind=policy_params.start_timestep, first_log=False)
    time_logger = TimeLogger()
    return [policy_params, env_name, max_goal, action_dim, goal_dim, max_action, expl_noise_std_l, expl_noise_std_h,
            c, episode_len, max_timestep, start_timestep, batch_size,
            log_interval, checkpoint_interval, evaluation_interval, save_video, video_interval, env, video_log_trigger, state_print_trigger, checkpoint_logger, evalutil_logger, time_logger,
            sample_n]


def initialize_params_checkpoint(params, device):
    policy_params = params.policy_params
    env_name = params.env_name
    max_goal = Tensor(policy_params.max_goal).to(device)
    action_dim = params.action_dim
    goal_dim = params.goal_dim
    max_action = policy_params.max_action
    expl_noise_std_l = policy_params.expl_noise_std_l
    expl_noise_std_h = policy_params.expl_noise_std_h
    c = policy_params.c
    episode_len = policy_params.episode_len
    max_timestep = policy_params.max_timestep
    start_timestep = policy_params.start_timestep
    batch_size = policy_params.batch_size
    sample_n = policy_params.sample_n
    log_interval = params.log_interval
    checkpoint_interval = params.checkpoint_interval
    save_video = params.save_video
    video_interval = params.video_interval
    env = get_env(params.env_name)
    return [policy_params, env_name, max_goal, action_dim, goal_dim, max_action, expl_noise_std_l, expl_noise_std_h,
            c, episode_len, max_timestep, start_timestep, batch_size,
            log_interval, checkpoint_interval, save_video, video_interval, env, sample_n]


def record_logger(args, option, step):
    if option == "inter_loss":
        target_q_l, critic_loss_l, actor_loss_l, target_q_h, critic_loss_h, actor_loss_h = args[:]
        if target_q_l is not None: wandb.log({'target_q low': torch.mean(target_q_l).squeeze()}, step=step)
        if critic_loss_l is not None: wandb.log({'critic_loss low': torch.mean(critic_loss_l).squeeze()}, step=step)
        if actor_loss_l is not None: wandb.log({'actor_loss low': torch.mean(actor_loss_l).squeeze()}, step=step)
        if target_q_h is not None: wandb.log({'target_q high': torch.mean(target_q_h).squeeze()}, step=step)
        if critic_loss_h is not None: wandb.log({'critic_loss high': torch.mean(critic_loss_h).squeeze()}, step=step)
        if actor_loss_h is not None: wandb.log({'actor_loss high': torch.mean(actor_loss_h).squeeze()}, step=step)
    elif option == "reward":
        episode_reward_l, episode_reward_h = args[:]
        wandb.log({'episode reward low': episode_reward_l}, step=step)
        wandb.log({'episode reward high': episode_reward_h}, step=step)
    elif option == "success_rate":
        success_rate = args[0]
        wandb.log({'success rate': success_rate}, step=step)


def create_rl_components(params, device, sample_n):
    # function local utils
    policy_params = params.policy_params
    state_dim, goal_dim, action_dim = params.state_dim, params.goal_dim, params.action_dim
    goal_dim = (goal_dim * sample_n)
    max_goal = Tensor(policy_params.max_goal)
    # low-level
    step, episode_num_h = 0, 0
    actor_eval_l = ActorLow(state_dim, goal_dim, action_dim, policy_params.max_action).to(device)
    actor_target_l = copy.deepcopy(actor_eval_l).to(device)
    actor_optimizer_l = torch.optim.Adam(actor_eval_l.parameters(), lr=policy_params.actor_lr)
    critic_eval_l = CriticLow(state_dim, goal_dim, action_dim).to(device)
    critic_target_l = copy.deepcopy(critic_eval_l).to(device)
    critic_optimizer_l = torch.optim.Adam(critic_eval_l.parameters(), lr=policy_params.critic_lr)
    experience_buffer_l = ExperienceBufferLow(policy_params.max_timestep, state_dim, goal_dim, action_dim, params.use_cuda)
    # high-level
    actor_eval_2 = ActorHigh(state_dim, goal_dim, max_goal, device).to(device)
    actor_target_2 = copy.deepcopy(actor_eval_2).to(device)
    actor_optimizer_2 = torch.optim.Adam(actor_eval_2.parameters(), lr=policy_params.actor_lr)
    critic_eval_2 = CriticHigh(state_dim, goal_dim).to(device)
    critic_target_2 = copy.deepcopy(critic_eval_2).to(device)
    critic_optimizer_2 = torch.optim.Adam(critic_eval_2.parameters(), lr=policy_params.critic_lr)
    experience_buffer_2 = ExperienceBufferHigh(int(policy_params.max_timestep / policy_params.c) + 1, state_dim, goal_dim, params.use_cuda)

    actor_eval_3 = ActorHigh(state_dim, goal_dim, max_goal, device).to(device)
    actor_target_3 = copy.deepcopy(actor_eval_3).to(device)
    actor_optimizer_3 = torch.optim.Adam(actor_eval_3.parameters(), lr=policy_params.actor_lr)
    critic_eval_3 = CriticHigh(state_dim, goal_dim).to(device)
    critic_target_3 = copy.deepcopy(critic_eval_3).to(device)
    critic_optimizer_3 = torch.optim.Adam(critic_eval_3.parameters(), lr=policy_params.critic_lr)
    experience_buffer_3 = ExperienceBufferHigh(int(policy_params.max_timestep / policy_params.c) + 1, state_dim, goal_dim, params.use_cuda)

    actor_eval_4 = ActorHigh(state_dim, goal_dim, max_goal, device).to(device)
    actor_target_4 = copy.deepcopy(actor_eval_4).to(device)
    actor_optimizer_4 = torch.optim.Adam(actor_eval_4.parameters(), lr=policy_params.actor_lr)
    critic_eval_4 = CriticHigh(state_dim, goal_dim).to(device)
    critic_target_4 = copy.deepcopy(critic_eval_4).to(device)
    critic_optimizer_4 = torch.optim.Adam(critic_eval_4.parameters(), lr=policy_params.critic_lr)
    experience_buffer_4 = ExperienceBufferHigh(int(policy_params.max_timestep / policy_params.c) + 1, state_dim, goal_dim, params.use_cuda)

    actor_eval_5 = ActorHigh(state_dim, goal_dim, max_goal, device).to(device)
    actor_target_5 = copy.deepcopy(actor_eval_5).to(device)
    actor_optimizer_5 = torch.optim.Adam(actor_eval_5.parameters(), lr=policy_params.actor_lr)
    critic_eval_5 = CriticHigh(state_dim, goal_dim).to(device)
    critic_target_5 = copy.deepcopy(critic_eval_5).to(device)
    critic_optimizer_5 = torch.optim.Adam(critic_eval_5.parameters(), lr=policy_params.critic_lr)
    experience_buffer_5 = ExperienceBufferHigh(int(policy_params.max_timestep / policy_params.c) + 1, state_dim, goal_dim, params.use_cuda)

    policy_network = Policy_Network(state_dim, 50, 1, device)

    return [step, episode_num_h,
            actor_eval_l, actor_target_l, actor_optimizer_l, critic_eval_l, critic_target_l, critic_optimizer_l, experience_buffer_l,
            actor_eval_2, actor_target_2, actor_optimizer_2, critic_eval_2, critic_target_2, critic_optimizer_2, experience_buffer_2,
            actor_eval_3, actor_target_3, actor_optimizer_3, critic_eval_3, critic_target_3, critic_optimizer_3, experience_buffer_3,
            actor_eval_4, actor_target_4, actor_optimizer_4, critic_eval_4, critic_target_4, critic_optimizer_4, experience_buffer_4,
            actor_eval_5, actor_target_5, actor_optimizer_5, critic_eval_5, critic_target_5, critic_optimizer_5, experience_buffer_5,
            policy_network]


def h_function(state, goal, next_state, goal_dim, sample_n):
    # return next goal
    #pdb.set_trace()
    return state[:goal_dim].repeat(1,sample_n) + goal - next_state[:goal_dim].repeat(1,sample_n)


def intrinsic_reward(state, goal, next_state):
    # low-level dense reward (L2 norm), provided by high-level policy
    return -torch.pow(sum(torch.pow(state + goal - next_state, 2)), 1 / 2)


def intrinsic_reward_simple(state, goal, next_state, goal_dim, sample_n):
    # low-level dense reward (L2 norm), provided by high-level policy
    #for i in range(sample_n)
    intri_reward_list = []
    for i in range(sample_n): intri_reward_list.append(-torch.pow(sum(torch.pow(state[:goal_dim] + goal[i * goal_dim:(i + 1) * goal_dim] - next_state[:3], 2)), 1 / 2))
    return sum(intri_reward_list)


def dense_reward(state, goal_dim, target=Tensor([0, 19, 0.5])):
    device = state.device
    target = target.to(device)
    l2_norm = torch.pow(sum(torch.pow(state[:goal_dim] - target, 2)), 1 / 2)
    return -l2_norm


def done_judge_low(goal):
    # define low-level success: same as high-level success (L2 norm < 5, paper B.2.2)
    l2_norm = torch.pow(sum(torch.pow(goal, 2)), 1 / 2)
    # done = (l2_norm <= 5.)
    done = (l2_norm <= 1.5)
    return Tensor([done])


def success_judge(state, goal_dim, target=Tensor([0, 19, 0.5])):
    location = state[:goal_dim]
    l2_norm = torch.pow(sum(torch.pow(location - target, 2)), 1 / 2)
    done = (l2_norm <= 5.)
    return Tensor([done])


def off_policy_correction(actor, action_sequence, state_sequence, goal_dim, goal, end_state, max_goal, device, sample_n):
    # initialize
    action_sequence = torch.stack(action_sequence).to(device)
    state_sequence = torch.stack(state_sequence).to(device)
    max_goal = max_goal.cpu()
    # prepare candidates
    mean = (end_state - state_sequence[0])[:goal_dim].cpu()
    std = 0.5 * max_goal
    candidates = [torch.min(torch.max(Tensor(np.random.normal(loc=mean, scale=std, size=goal_dim).astype(np.float32)), -max_goal), max_goal).repeat(sample_n) for _ in range(8)]
    candidates.append(mean.repeat(sample_n))
    candidates.append(goal.cpu())
    # select maximal
    candidates = torch.stack(candidates).to(device)
    surr_prob = [-functional.mse_loss(action_sequence, actor(state_sequence, state_sequence[0][:goal_dim].repeat(sample_n)  + candidate - state_sequence[:, :goal_dim].repeat(1,sample_n))).detach().numpy() for candidate in candidates]
    #pdb.set_trace()
    index = int(np.argmax(surr_prob))
    updated = (index != 9)
    goal_hat = candidates[index]
    return goal_hat.cpu(), updated


def step_update_l(experience_buffer, batch_size, total_it, actor_eval, actor_target, critic_eval, critic_target, critic_optimizer, actor_optimizer, params):
    # initialize
    policy_params = params.policy_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    total_it[0] += 1
    # sample mini-batch transitions
    state, goal, action, reward, next_state, next_goal, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        policy_noise = Tensor(np.random.normal(loc=0, scale=policy_params.policy_noise_std, size=params.action_dim).astype(np.float32) * policy_params.policy_noise_scale) \
            .clamp(-policy_params.policy_noise_clip, policy_params.policy_noise_clip).to(device)
        next_action = (actor_target(next_state, next_goal) + policy_noise).clamp(-policy_params.max_action, policy_params.max_action)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(next_state, next_goal, next_action)
        q_target = torch.min(q_target_1, q_target_2)
        y = policy_params.reward_scal_l * reward + (1 - done) * policy_params.discount * q_target
    # update critic q_evaluate
    q_eval_1, q_eval_2 = critic_eval(state, goal, action)
    critic_loss = functional.mse_loss(q_eval_1, y) + functional.mse_loss(q_eval_2, y)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # delayed policy update
    actor_loss = None
    if total_it[0] % policy_params.policy_freq == 0:
        # compute actor loss
        actor_loss = -critic_eval.q1(state, goal, actor_eval(state, goal)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # soft update: critic q_target
        for param_eval, param_target in zip(critic_eval.parameters(), critic_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        for param_eval, param_target in zip(actor_eval.parameters(), actor_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        actor_loss = actor_loss.detach()
    return y.detach(), critic_loss.detach(), actor_loss


def step_update_h(experience_buffer, batch_size, total_it, actor_eval, actor_target, critic_eval, critic_target, critic_optimizer, actor_optimizer, params):
    policy_params = params.policy_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    max_goal = Tensor(policy_params.max_goal).to(device)
    # sample mini-batch transitions
    state_start, goal, reward, state_end, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        policy_noise = Tensor(np.random.normal(loc=0, scale=policy_params.policy_noise_std, size=params.goal_dim).astype(np.float32) * policy_params.policy_noise_scale) \
            .clamp(-policy_params.policy_noise_clip, policy_params.policy_noise_clip).to(device)
        next_goal = torch.min(torch.max(actor_target(state_end) + policy_noise, -max_goal), max_goal)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(state_end, next_goal)
        q_target = torch.min(q_target_1, q_target_2)
        y = policy_params.reward_scal_h * reward + (1 - done) * policy_params.discount * q_target
    # update critic q_evaluate
    q_eval_1, q_eval_2 = critic_eval(state_start, goal)
    critic_loss = functional.mse_loss(q_eval_1, y) + functional.mse_loss(q_eval_2, y)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # delayed policy updates
    actor_loss = None
    if int(total_it[0] / policy_params.c) % policy_params.policy_freq == 0:
        # compute actor loss
        actor_loss = -critic_eval.q1(state_start, actor_eval(state_start)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # soft update: critic q_target
        for param_eval, param_target in zip(critic_eval.parameters(), critic_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        for param_eval, param_target in zip(actor_eval.parameters(), actor_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        actor_loss = actor_loss.detach()
    return y.detach(), critic_loss.detach(), actor_loss


def evaluate(actor_l, actor_h, params, target_pos, device):
    policy_params = params.policy_params
    print("\n    > evaluating policies...")
    success_number = 0
    env = get_env(params.env_name)
    goal_dim = params.goal_dim
    for i in range(10):
        env.seed(policy_params.seed + i)
        for j in range(5):
            t = 0
            episode_len = policy_params.episode_len
            obs, done = Tensor(env.reset()).to(device), False
            goal = Tensor(torch.randn(goal_dim)).to(device)
            while not done and t < episode_len:
                t += 1
                action = actor_l(obs, goal).to(device)
                obs, _, _, _ = env.step(action.detach().cpu())
                obs = Tensor(obs).to(device)
                done = success_judge(obs, goal_dim, target_pos)
                goal = actor_h(obs)
            if done:
                success_number += 1
        print("        > evaluated {} episodes".format(i * 5 + j + 1))
    success_rate = success_number / 50
    print("    > finished evaluation, success rate: {}\n".format(success_rate))
    return success_rate

def high_level_add_update(t,c,start_timestep, expl_noise_std_h, device, episode_reward_h, done_h, episode_timestep_l, episode_reward_l, episode_num_l,
                          batch_size, total_it, goal, next_state,
                          actor_target_l,actor_eval_h, experience_buffer_h, actor_target_h, critic_eval_h, critic_target_h, critic_optimizer_h, actor_optimizer_h,
                          state_sequence, action_sequence, goal_sequence,
                          max_goal, params, goal_dim, sample_n):
    next_goal = goal
    if (t + 1) % c == 0 and t > 0:
        # 2.2.7 sample goal
        if t < start_timestep:
            next_goal = (torch.randn_like(goal) * max_goal)
            next_goal = torch.min(torch.max(next_goal, -max_goal), max_goal)
        else:
            expl_noise_goal = np.random.normal(loc=0, scale=expl_noise_std_h, size=goal_dim).astype(np.float32)
            next_goal = (actor_eval_h(next_state.to(device)).detach().cpu() + expl_noise_goal).squeeze().to(device)
            next_goal = torch.min(torch.max(next_goal, -max_goal), max_goal)
        # 2.2.8 collect high-level experience
        goal_hat, updated = off_policy_correction(actor_target_l, action_sequence, state_sequence, goal_dim,
                                                  goal_sequence[0], next_state, max_goal, device, sample_n)
        experience_buffer_h.add(state_sequence[0], goal_hat, episode_reward_h, next_state, done_h)
        # if state_print_trigger.good2log(t, 500): print_cmd_hint(params=[state_sequence, goal_sequence, action_sequence, intri_reward_sequence, updated, goal_hat, reward_h_sequence], location='training_state')
        # 2.2.9 reset segment arguments & log (reward)
        state_sequence, action_sequence, intri_reward_sequence, goal_sequence, reward_h_sequence = [], [], [], [], []
        print(f"    > Segment: Total T: {t + 1} Episode_L Num: {episode_num_l + 1} Episode_L T: {episode_timestep_l} Reward_L: {float(episode_reward_l):.3f} Reward_H: {float(episode_reward_h):.3f}")
        if t >= start_timestep: record_logger(args=[episode_reward_l, episode_reward_h], option='reward',
                                              step=t - start_timestep)
        episode_reward_l, episode_timestep_l = 0, 0
        episode_reward_h = 0
        episode_num_l += 1

        if t >= start_timestep and (t + 1) % c == 0:
            target_q_h, critic_loss_h, actor_loss_h = \
                step_update_h(experience_buffer_h, batch_size, total_it, actor_eval_h, actor_target_h, critic_eval_h, critic_target_h, critic_optimizer_h, actor_optimizer_h, params)

    return next_goal, experience_buffer_h


def train(params):
    # 1. Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    if params.checkpoint is None:
        # > rl components
        [step, episode_num_h,
         actor_eval_l, actor_target_l, actor_optimizer_l, critic_eval_l, critic_target_l, critic_optimizer_l, experience_buffer_l,
         actor_eval_2, actor_target_2, actor_optimizer_2, critic_eval_2, critic_target_2, critic_optimizer_2, experience_buffer_2,
         actor_eval_3, actor_target_3, actor_optimizer_3, critic_eval_3, critic_target_3, critic_optimizer_3, experience_buffer_3,
         actor_eval_4, actor_target_4, actor_optimizer_4, critic_eval_4, critic_target_4, critic_optimizer_4, experience_buffer_4,
         actor_eval_5, actor_target_5, actor_optimizer_5, critic_eval_5, critic_target_5, critic_optimizer_5, experience_buffer_5,
         policy_network] = create_rl_components(params, device, params.policy_params.sample_n)
        # > running utils
        [policy_params, env_name, max_goal, action_dim, goal_dim, max_action, expl_noise_std_l, expl_noise_std_h,
         c, episode_len, max_timestep, start_timestep, batch_size,
         log_interval, checkpoint_interval, evaluation_interval, save_video, video_interval, env, video_log_trigger, state_print_trigger, checkpoint_logger, evalutil_logger, time_logger, sample_n] = initialize_params(params, device)
    else:
        # > rl components
        prefix = params.prefix
        [step, params, device, [time_logger, state_print_trigger, video_log_trigger, checkpoint_logger, evalutil_logger, episode_num_h],
         actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,
         actor_eval_2, actor_target_2, actor_optimizer_2, critic_eval_2, critic_target_2, critic_optimizer_2, experience_buffer_2,
         actor_eval_3, actor_target_3, actor_optimizer_3, critic_eval_3, critic_target_3, critic_optimizer_3, experience_buffer_3,
         actor_eval_4, actor_target_4, actor_optimizer_4, critic_eval_4, critic_target_4, critic_optimizer_4, experience_buffer_4,
         actor_eval_5, actor_target_5, actor_optimizer_5, critic_eval_5, critic_target_5, critic_optimizer_5,  experience_buffer_5,
         policy_network] = load_checkpoint(params.checkpoint)
        # > running utils
        [policy_params, env_name, max_goal, action_dim, goal_dim, max_action, expl_noise_std_l, expl_noise_std_h,
         c, episode_len, max_timestep, start_timestep, batch_size,
         log_interval, checkpoint_interval, evaluation_interval, save_video, video_interval, env, sample_n] = initialize_params_checkpoint(params, device)
        params.prefix = prefix
    target_q_2, critic_loss_2, actor_loss_2 = None, None, None
    target_q_3, critic_loss_3, actor_loss_3 = None, None, None
    target_q_4, critic_loss_4, actor_loss_4 = None, None, None
    target_q_5, critic_loss_5, actor_loss_5 = None, None, None

    target_pos = get_target_position(env_name).to(device)
    # 1.2 set seeds
    env.seed(policy_params.seed)
    torch.manual_seed(policy_params.seed)
    np.random.seed(policy_params.seed)

    # 2. Training Algorithm (TD3)
    # 2.1 initialize
    print_cmd_hint(params=params, location='start_train')
    time_logger.time_spent()
    total_it = [0]

    success_rate, episode_reward_l, episode_reward, episode_num_l, episode_timestep_l = 0, 0, 0, 0, 1
    episode_reward_h, episode_timestep_h = 0, 1

    state = Tensor(env.reset()).to(device)
    goal2 = Tensor(torch.randn(goal_dim)).to(device)
    goal3 = Tensor(torch.randn(goal_dim)).to(device)
    goal4 = Tensor(torch.randn(goal_dim)).to(device)
    goal5 = Tensor(torch.randn(goal_dim)).to(device)
    goal = Tensor(torch.randn(goal_dim * sample_n)).to(device)
    state_sequence2, goal_sequence2, action_sequence2, intri_reward_sequence2, reward_h_sequence2 = [], [], [], [], []
    state_sequence3, goal_sequence3, action_sequence3, intri_reward_sequence3, reward_h_sequence3 = [], [], [], [], []
    state_sequence4, goal_sequence4, action_sequence4, intri_reward_sequence4, reward_h_sequence4 = [], [], [], [], []
    state_sequence5, goal_sequence5, action_sequence5, intri_reward_sequence5, reward_h_sequence5 = [], [], [], [], []

    hidden_policy_network = init_hidden(3, 300 * 4 * 31, device=device, grad=True)
    masks = [torch.ones(3, 1).to(device) for _ in range(2 * 1 + 1)]

    # 2.2 training loop
    for t in range(step, max_timestep):
        # 2.2.1 sample action
        if t < start_timestep:
            action = env.action_space.sample()
        else:
            pdb.set_trace()
            expl_noise_action = np.random.normal(loc=0, scale=expl_noise_std_l, size=action_dim).astype(np.float32)
            action = (actor_eval_l(state, goal).detach().cpu() + expl_noise_action).clamp(-max_action, max_action).squeeze()
        # 2.2.2 interact environment
        next_state, _, _, info = env.step(action)
        # 2.2.3 compute step arguments
        reward_h = dense_reward(state, goal_dim, target=target_pos)
        done_h = success_judge(state, goal_dim, target_pos)
        next_state, action, reward_h, done_h = Tensor(next_state).to(device), Tensor(action), Tensor([reward_h]), Tensor([done_h])
        intri_reward = intrinsic_reward_simple(state, goal, next_state, goal_dim, sample_n)
        next_goal = h_function(state, goal, next_state, goal_dim, sample_n)
        done_l = done_judge_low(goal)
        # 2.2.4 collect low-level experience
        experience_buffer_l.add(state, goal, action, intri_reward, next_state, next_goal, done_l)
        # 2.2.5 record segment arguments
        state_sequence2.append(state)
        state_sequence3.append(state)
        state_sequence4.append(state)
        state_sequence5.append(state)
        action_sequence2.append(action)
        action_sequence3.append(action)
        action_sequence4.append(action)
        action_sequence5.append(action)
        intri_reward_sequence2.append(intri_reward)
        intri_reward_sequence3.append(intri_reward)
        intri_reward_sequence4.append(intri_reward)
        intri_reward_sequence5.append(intri_reward)
        goal_sequence2.append(goal)
        goal_sequence3.append(goal)
        goal_sequence4.append(goal)
        goal_sequence5.append(goal)
        reward_h_sequence2.append(reward_h)
        reward_h_sequence3.append(reward_h)
        reward_h_sequence4.append(reward_h)
        reward_h_sequence5.append(reward_h)
        # 2.2.6 update low-level segment reward
        episode_reward_l += intri_reward
        episode_reward_h += reward_h
        episode_reward += reward_h

        next_goal2, experience_buffer_2 = high_level_add_update(t, c, start_timestep, expl_noise_std_h, device,
                                                                episode_reward_h, done_h, episode_timestep_l,
                                                                episode_reward_l, episode_num_l,
                                                                batch_size, total_it, goal2, next_state, actor_target_l,
                                                                actor_eval_2, experience_buffer_2, actor_target_2,
                                                                critic_eval_2, critic_target_2, critic_optimizer_2,
                                                                actor_optimizer_2,
                                                                state_sequence2, action_sequence2, goal_sequence2,
                                                                max_goal, params, goal_dim, sample_n)

        next_goal3, experience_buffer_3 = high_level_add_update(t, c, start_timestep, expl_noise_std_h, device,
                                                                episode_reward_h, done_h, episode_timestep_l,
                                                                episode_reward_l, episode_num_l,
                                                                batch_size, total_it, goal3, next_state, actor_target_l,
                                                                actor_eval_3, experience_buffer_3, actor_target_3,
                                                                critic_eval_3, critic_target_3, critic_optimizer_3,
                                                                actor_optimizer_3,
                                                                state_sequence3, action_sequence3, goal_sequence3,
                                                                max_goal, params, goal_dim, sample_n)

        next_goal4, experience_buffer_4 = high_level_add_update(t, c, start_timestep, expl_noise_std_h, device,
                                                                episode_reward_h, done_h, episode_timestep_l,
                                                                episode_reward_l, episode_num_l,
                                                                batch_size, total_it, goal4, next_state, actor_target_l,
                                                                actor_eval_4, experience_buffer_4, actor_target_4,
                                                                critic_eval_4, critic_target_4, critic_optimizer_4,
                                                                actor_optimizer_4,
                                                                state_sequence4, action_sequence4, goal_sequence4,
                                                                max_goal, params, goal_dim, sample_n)

        next_goal5, experience_buffer_5 = high_level_add_update(t, c, start_timestep, expl_noise_std_h, device,
                                                                episode_reward_h, done_h, episode_timestep_l,
                                                                episode_reward_l, episode_num_l,
                                                                batch_size, total_it, goal5, next_state, actor_target_l,
                                                                actor_eval_5, experience_buffer_5, actor_target_5,
                                                                critic_eval_5, critic_target_5, critic_optimizer_5,
                                                                actor_optimizer_5,
                                                                state_sequence5, action_sequence5, goal_sequence5,
                                                                max_goal, params, goal_dim, sample_n)

        hierarchies_selected, hidden_policy_network = policy_network(state, next_goal5, next_goal4, next_goal3, hidden_policy_network, masks[-1])
        hierarchies_selected = torch.cat([hierarchies_selected, torch.ones((1, 1)).type(torch.float)], dim=0)
        hierarchies_selected = (hierarchies_selected / hierarchies_selected.sum())

        #n = 2
        tensors = [next_goal5, next_goal4, next_goal3, next_goal2]
        sampled_indices = torch.multinomial(hierarchies_selected.reshape((1,4)), num_samples=sample_n, replacement=False)
        #sampled_tensors = [tensors[i] for i in sampled_indices[0]]
        #for i in sampled_indices[0]: tensors[i]
        next_goal = torch.cat([tensors[i] for i in sampled_indices[0]])

        # 2.2.10 update observations
        state = next_state
        goal = next_goal

        # 2.2.11 update networks
        if t >= start_timestep:
            target_q_l, critic_loss_l, actor_loss_l = \
                step_update_l(experience_buffer_l, batch_size, total_it, actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, critic_optimizer_l, actor_optimizer_l, params)

        # 2.2.12 log training curve (inter_loss)
        #if t >= start_timestep and t % log_interval == 0:
        #    record_logger(args=[target_q_l, critic_loss_l, actor_loss_l, target_q_h, critic_loss_h, actor_loss_h], option='inter_loss', step=t-start_timestep)
        #    record_logger([success_rate], 'success_rate', step=t - start_timestep)
        # 2.2.13 start new episode
        if episode_timestep_h >= episode_len:
            # > update loggers
            if t > start_timestep: episode_num_h += 1
            else: episode_num_h = 0
            print(f"    >>> Episode: Total T: {t + 1} Episode_H Num: {episode_num_h+1} Episode_H T: {episode_timestep_h} Reward_Episode: {float(episode_reward):.3f}\n")
            # > clear loggers
            episode_reward = 0
            state_sequence2, goal_sequence2, action_sequence2, intri_reward_sequence2, reward_h_sequence2 = [], [], [], [], []
            state_sequence3, goal_sequence3, action_sequence3, intri_reward_sequence3, reward_h_sequence3 = [], [], [], [], []
            state_sequence4, goal_sequence4, action_sequence4, intri_reward_sequence4, reward_h_sequence4 = [], [], [], [], []
            state_sequence5, goal_sequence5, action_sequence5, intri_reward_sequence5, reward_h_sequence5 = [], [], [], [], []
            episode_reward_l, episode_timestep_l, episode_num_l = 0, 0, 0
            state, done_h = Tensor(env.reset()).to(device), Tensor([False])
            episode_reward_h, episode_timestep_h = 0, 0
        # 2.2.14 update training loop arguments
        episode_timestep_l += 1
        episode_timestep_h += 1
        # 2.2.15 save videos & checkpoints
        '''
        if save_video and video_log_trigger.good2log(t, video_interval):
            log_video_hrl(env_name, actor_target_l, actor_target_h, params)
            time_logger.sps(t)
            time_logger.time_spent()
            print("")
        if checkpoint_logger.good2log(t, checkpoint_interval):
            logger = [time_logger, state_print_trigger, video_log_trigger, checkpoint_logger, evalutil_logger, episode_num_h]
            save_checkpoint(t, actor_target_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,
                            actor_target_h, critic_target_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h,
                            logger, params)
        if t > start_timestep and evalutil_logger.good2log(t, evaluation_interval):
            success_rate = evaluate(actor_target_l, actor_target_h, params, target_pos, device)
    # 2.3 final log (episode videos)
    logger = [time_logger, state_print_trigger, video_log_trigger, checkpoint_logger, evalutil_logger, episode_num_h]
    save_checkpoint(max_timestep, actor_target_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,
                    actor_target_h, critic_target_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h,
                    logger, params)
    for i in range(3):
        log_video_hrl(env_name, actor_target_l, actor_target_h, params)
    print_cmd_hint(params=params, location='end_train')'''


if __name__ == "__main__":
    env_name = "AntFall"
    env = get_env(env_name)
    state_dim = env.observation_space.shape[0]
    goal_dim = 3
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_goal = [10., 10., .5]
    policy_params = ParamDict(
        seed=54321,
        c=10,
        policy_noise_scale=0.2,
        policy_noise_std=1.,
        expl_noise_std_l=1.,
        expl_noise_std_h=1.,
        policy_noise_clip=0.5,
        max_action=max_action,
        max_goal=max_goal,
        discount=0.99,
        policy_freq=1,
        tau=5e-3,
        actor_lr=1e-4,
        critic_lr=1e-3,
        reward_scal_l=1.,
        reward_scal_h=.1,
        episode_len=1000,
        max_timestep=int(3e6),
        start_timestep=int(300),
        batch_size=100,
        sample_n=2
    )
    params = ParamDict(
        policy_params=policy_params,
        env_name=env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=3,
        video_interval=int(1e4),
        log_interval=5,
        checkpoint_interval=int(1e5),
        evaluation_interval=int(1e4),
        prefix="test_simple_origGoal_fixedIntriR_posER",
        save_video=True,
        use_cuda=True,
        # checkpoint="hiro-antpush_test_simple_origGoal_fixedIntriR_posER-it(2000000)-[2020-07-02 20:35:25.673267].tar"
        checkpoint=None
    )

    wandb.init(project="ziang-hiro-new")
    train(params=params)
