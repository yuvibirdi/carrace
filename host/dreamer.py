"""Dreamer-style training: learn a model of the car from the recordings, then improve
the driving policy by practising inside that model, with "get round the lap faster"
as the reward.

    python3 host/dreamer.py [model-steps] [policy-steps] [speed-cap] [data-dir]

What is learned, all from the recorded episodes (any speed, any driver, human or network):
  world model  given what the car has seen and the action taken, predict what it sees
               next, how much of the lap that step covered (the reward), and whether
               the lap is over
  actor        picks steering and a throttle level for each moment; it starts as a copy
               of the recorded driving and is then pushed towards more lap progress
               per step in imagined drives, while a copy-the-driver term keeps it close
               to situations the world model has actually seen
  critic       estimates the progress still to come, so short imagined drives can be judged

The pictures are not modelled directly. The convolution layers of the already trained
driving network (data/model.pt) turn each frame into features, and the world model
predicts those. That is the pretrained part.

speed-cap is the curriculum: the actor may only choose throttle levels up to it. Raise
it one level at a time, each time after the car laps cleanly at the current one and
those laps have been recorded and added to the data.
"""
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal, kl_divergence

import episodes
from model import H, W, Net, prepare, steer_class

SPEEDS = [0, 60, 70, 80, 90, 100]  # throttle levels (percent) the actor chooses between
ACTIONS = 3 + len(SPEEDS)  # steering one-hot followed by throttle-level one-hot
SEQ, BATCH, HORIZON = 32, 32, 15  # training sequence length, batch size, imagined steps
DETER, STOCH, HIDDEN = 256, 32, 256
GAMMA, LAMBDA = 0.99, 0.95
REWARD_SCALE = 100  # reward = percent of the lap covered in that step
FAIL_PENALTY = 25  # a run marked as failed (F on the teleop page) ends with minus this


def load(folder, device):
    """Every usable episode as tensors: features and actions. Rewards come from label()."""
    bc = Net().to(device)
    bc.load_state_dict(torch.load(folder / "model.pt", map_location=device)["net"])
    bc.eval()
    result = []
    for path in sorted(folder.glob("ep_*.mjpeg")) + sorted((folder / "other").glob("ep_*.mjpeg")):
        rows = list(episodes.read(path))
        moving = [i for i, r in enumerate(rows) if r[1]]
        if len(moving) < 2 or moving[-1] - moving[0] < SEQ + 1:
            continue
        rows = rows[moving[0] : moving[-1] + 1]
        frames = torch.from_numpy(np.stack([prepare(r[0]) for r in rows])).to(device)
        with torch.no_grad():
            feat = bc.features(frames.permute(0, 3, 1, 2).float() / 255 - 0.5)
        steer = torch.tensor([steer_class(r[2], r[3]) for r in rows], device=device)
        level = torch.tensor(
            [min(range(len(SPEEDS)), key=lambda k: abs(SPEEDS[k] - r[1])) for r in rows], device=device
        )
        action = torch.cat([F.one_hot(steer, 3), F.one_hot(level, len(SPEEDS))], 1).float()
        result.append(dict(name=path.name, feat=feat, steer=steer, level=level, action=action))
    if not result:
        sys.exit(f"no usable episodes in {folder}/")
    return result


def label(eps):
    """Adds reward, lap-not-over flag and a worth-copying flag to every episode.

    A completed lap runs from 0 to 100 % progress evenly over its frames, so a faster lap
    earns more per frame. A failed run has no finish line, so its progress is read off
    the picture by a regression fitted on the completed laps, and it ends with a penalty."""
    done = [e for e in eps if "_fail" not in e["name"]]
    if not done:
        sys.exit("need at least one completed lap")
    for e in done:
        e["progress"] = torch.linspace(0, 1, len(e["steer"]), device=e["feat"].device)
    x, y = torch.cat([e["feat"] for e in done]), torch.cat([e["progress"] for e in done])
    gram = x @ x.T
    weights = torch.linalg.solve(gram + gram.diagonal().mean() * torch.eye(len(x), device=x.device), y - 0.5)
    for e in eps:
        failed = "_fail" in e["name"]
        if failed:
            # Measured on laps left out of the fit, this is off by about 0.1 of a lap.
            e["progress"] = torch.cummax((e["feat"] @ x.T @ weights + 0.5).clamp(0, 1), 0).values
        e["reward"] = torch.zeros_like(e["progress"])
        e["reward"][1:] = (e["progress"][1:] - e["progress"][:-1]) * REWARD_SCALE
        e["reward"][-1] -= FAIL_PENALTY * failed
        e["cont"] = torch.ones_like(e["progress"])
        e["cont"][-1] = 0  # the lap, or the failed run, ends here
        e["good"] = torch.full_like(e["progress"], float(not failed))


def sample(eps, batch=BATCH):
    """Random stretches of SEQ consecutive frames, stacked to [batch, SEQ, ...]."""
    weights = torch.tensor([float(len(e["steer"])) for e in eps])
    picks = []
    for i in torch.multinomial(weights, batch, replacement=True).tolist():
        start = np.random.randint(0, len(eps[i]["steer"]) - SEQ + 1)
        picks.append({k: v[start : start + SEQ] for k, v in eps[i].items() if k != "name"})  # all tensors
    return {k: torch.stack([p[k] for p in picks]) for k in picks[0]}


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, outputs))


class World(nn.Module):
    """Recurrent state-space model. The state is (h, z): h is a memory carried forward
    by a GRU, z is a small random part that is corrected by each new observation."""

    def __init__(self, feat_dim):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(feat_dim, HIDDEN), nn.ELU())
        self.inp = nn.Sequential(nn.Linear(STOCH + ACTIONS, HIDDEN), nn.ELU())
        self.gru = nn.GRUCell(HIDDEN, DETER)
        self.prior = mlp(DETER, 2 * STOCH)  # guess z from memory alone (used when imagining)
        self.post = mlp(DETER + HIDDEN, 2 * STOCH)  # z from memory plus what was actually seen
        self.decoder = mlp(DETER + STOCH, feat_dim)
        self.reward = mlp(DETER + STOCH, 1)
        self.cont = mlp(DETER + STOCH, 1)

    @staticmethod
    def dist(out):
        mean, std = out.chunk(2, -1)
        return Normal(mean, F.softplus(std) + 0.1)

    def step(self, h, z, action):
        return self.gru(self.inp(torch.cat([z, action], -1)), h)

    def observe(self, feat, action):
        """Run over real sequences. Returns states [B, T, DETER + STOCH] and the KL loss
        that teaches the prior to predict what the observation will reveal."""
        embed = self.embed(feat)
        h = torch.zeros(feat.shape[0], DETER, device=feat.device)
        states, kl = [], 0
        for t in range(feat.shape[1]):
            if t:
                h = self.step(h, z, action[:, t - 1])
            prior, post = self.dist(self.prior(h)), self.dist(self.post(torch.cat([h, embed[:, t]], -1)))
            z = post.rsample()
            states.append(torch.cat([h, z], -1))
            frozen_post, frozen_prior = Normal(post.mean.detach(), post.stddev.detach()), Normal(prior.mean.detach(), prior.stddev.detach())
            kl = kl + 0.5 * kl_divergence(frozen_post, prior).sum(-1).clamp(min=1).mean()
            kl = kl + 0.1 * kl_divergence(post, frozen_prior).sum(-1).clamp(min=1).mean()
        return torch.stack(states, 1), kl / feat.shape[1]

    def imagine_step(self, state, action):
        h = self.step(state[..., :DETER], state[..., DETER:], action)
        return torch.cat([h, self.dist(self.prior(h)).sample()], -1)


class Actor(nn.Module):
    def __init__(self, cap):
        super().__init__()
        self.net = mlp(DETER + STOCH, ACTIONS)
        allowed = torch.tensor([0.0 if s <= cap else float("-inf") for s in SPEEDS])
        self.register_buffer("allowed", allowed)  # the speed curriculum

    def forward(self, state):
        out = self.net(state)
        return Categorical(logits=out[..., :3]), Categorical(logits=out[..., 3:] + self.allowed)


class Step(nn.Module):
    """What runs on the car for each camera frame (pi/drive.cpp, through ONNX Runtime):
    update the state from the frame and the previous action, then ask the actor.
    The random part of the state is replaced by its most likely value."""

    def __init__(self, features, mean, std, world, actor):
        super().__init__()
        self.features, self.world, self.actor = features, world, actor
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, frame, h, z, action, first):
        """first is 1 on the first frame of a run (nothing to carry forward yet), else 0."""
        feat = (self.features(frame.permute(0, 3, 1, 2).float() / 255 - 0.5) - self.mean) / self.std
        h = first * h + (1 - first) * self.world.step(h, z, action)
        z = self.world.post(torch.cat([h, self.world.embed(feat)], -1)).chunk(2, -1)[0]
        out = self.actor.net(torch.cat([h, z], -1))
        return out[:, :3], out[:, 3:] + self.actor.allowed.clamp(min=-1e9), h, z


def export(folder, world, actor, mean, std):
    bc = Net()
    bc.load_state_dict(torch.load(folder / "model.pt", map_location="cpu")["net"])
    step = Step(bc.features, mean.cpu(), std.cpu(), world.cpu(), actor.cpu()).eval()
    example = (
        torch.zeros(1, H, W, 3, dtype=torch.uint8), torch.zeros(1, DETER), torch.zeros(1, STOCH),
        torch.zeros(1, ACTIONS), torch.ones(1, 1),
    )
    torch.onnx.export(
        step, example, str(folder / "dreamer.onnx"), dynamo=False, opset_version=17,
        input_names=["frame", "h", "z", "action", "first"], output_names=["steer", "level", "h_next", "z_next"],
    )


def one_hot_action(steer, level):
    return torch.cat([F.one_hot(steer, 3), F.one_hot(level, len(SPEEDS))], -1).float()


def imagine(world, actor, critic, start):
    """Drive HORIZON steps inside the world model from each start state.
    Returns per-step log-probabilities, entropies, lambda-returns and values."""
    state, states, logps, entropies = start, [start], [], []
    for _ in range(HORIZON):
        steer, level = actor(state)
        a_steer, a_level = steer.sample(), level.sample()
        logps.append(steer.log_prob(a_steer) + level.log_prob(a_level))
        entropies.append(steer.entropy() + level.entropy())
        with torch.no_grad():
            state = world.imagine_step(state, one_hot_action(a_steer, a_level))
        states.append(state)
    states = torch.stack(states)  # [HORIZON + 1, N, state]
    with torch.no_grad():
        reward = world.reward(states[1:]).squeeze(-1)
        cont = torch.sigmoid(world.cont(states[1:]).squeeze(-1))
    value = critic(states).squeeze(-1)
    returns, future = [], value[-1].detach()
    for k in reversed(range(HORIZON)):
        future = reward[k] + GAMMA * cont[k] * ((1 - LAMBDA) * value[k + 1].detach() + LAMBDA * future)
        returns.append(future)
    return torch.stack(logps), torch.stack(entropies), torch.stack(returns[::-1]), value[:-1], reward


def evaluate(world, actor, critic, eps):
    """On episodes never trained on: how far ahead the world model can predict, and how
    the actor compares with the recorded driver."""
    world.eval()
    ahead_error = still_error = agree = frames = 0.0
    with torch.no_grad():
        for e in eps:
            states, _ = world.observe(e["feat"][None], e["action"][None])
            steer, level = actor(states[0])
            agree += (steer.probs.argmax(-1) == e["steer"]).float().sum().item()
            frames += len(e["steer"])
            # Predict HORIZON frames ahead with no observations, using the recorded actions.
            starts = range(8, len(e["steer"]) - HORIZON, HORIZON)
            for t in starts:
                state = states[0, t][None]
                for k in range(HORIZON):
                    state = world.imagine_step(state, e["action"][t + k][None])
                target = e["feat"][t + HORIZON]
                ahead_error += F.mse_loss(world.decoder(state)[0], target).item() / len(starts) / len(eps)
                still_error += F.mse_loss(e["feat"][t], target).item() / len(starts) / len(eps)
        start = torch.cat([world.observe(e["feat"][None], e["action"][None])[0][0, 8::8] for e in eps])
        reward = imagine(world, actor, critic, start)[4]
        recorded = torch.cat([e["reward"][1:] for e in eps]).mean().item()
    world.train()
    return (
        f"held-out: predicting {HORIZON} frames ahead, error {ahead_error:.3f} "
        f"(assuming nothing changes: {still_error:.3f}); actor steers like the driver on {agree / frames:.0%} of frames; "
        f"imagined progress {reward.mean().item():.3f} %/frame vs recorded {recorded:.3f}"
    )


def main(model_steps=3000, policy_steps=2000, cap=None, folder="data"):
    folder = pathlib.Path(folder)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    eps = load(folder, device)
    # Features are standardised so every one counts equally in the prediction loss.
    everything = torch.cat([e["feat"] for e in eps])
    mean, std = everything.mean(0), everything.std(0) + 1e-3
    for e in eps:
        e["feat"] = (e["feat"] - mean) / std
    label(eps)
    done = [e for e in eps if "_fail" not in e["name"]]
    held = done[4::5] if len(done) >= 5 else done[-1:]
    train = [e for e in eps if all(e is not h for h in held)] or eps
    fastest = max(SPEEDS[e["level"].max().item()] for e in eps)
    cap = cap or SPEEDS[min(SPEEDS.index(fastest) + 1, len(SPEEDS) - 1)]
    if cap not in SPEEDS[1:]:
        sys.exit(f"speed-cap must be one of {SPEEDS[1:]}")
    used = sorted({SPEEDS[k] for e in eps for k in e["level"].unique().tolist()})
    print(
        f"{len(train)} training ({len(eps) - len(done)} of them failed runs) and {len(held)} held-out episodes, "
        f"{sum(len(e['steer']) for e in eps)} frames "
        f"on {device}; throttle levels in the data {used}; actor may use up to {cap} %", flush=True,
    )

    world, actor, critic = World(everything.shape[1]).to(device), Actor(cap).to(device), mlp(DETER + STOCH, 1).to(device)
    world_opt = torch.optim.Adam(world.parameters(), lr=3e-4)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    log = open(folder / "dreamer_log.csv", "w")
    log.write("phase,step,predict,reward,kl,copy,imagined_return\n")

    for step in range(1, model_steps + 1):  # 1. the world model
        b = sample(train)
        states, kl = world.observe(b["feat"], b["action"])
        predict = F.mse_loss(world.decoder(states), b["feat"])
        reward = F.mse_loss(world.reward(states[:, 1:]).squeeze(-1), b["reward"][:, 1:])
        cont = F.binary_cross_entropy_with_logits(world.cont(states).squeeze(-1), b["cont"])
        world_opt.zero_grad()
        (predict + reward + cont + kl).backward()
        nn.utils.clip_grad_norm_(world.parameters(), 100)
        world_opt.step()
        if step % 250 == 0 or step == model_steps:
            log.write(f"model,{step},{predict.item():.4f},{reward.item():.5f},{kl.item():.4f},,\n")
            print(f"model {step:5d}  predict {predict.item():.3f}  reward {reward.item():.4f}  kl {kl.item():.2f}", flush=True)
    for p in world.parameters():
        p.requires_grad_(False)

    warmup = policy_steps // 4  # first only copy the recorded driving
    for step in range(1, policy_steps + 1):  # 2. the actor and critic, inside the world model
        b = sample(train)
        with torch.no_grad():
            states, _ = world.observe(b["feat"], b["action"])
        steer, level = actor(states)
        # Only completed laps are copied. Recorded throttle above the cap is copied as the cap itself.
        copy = -(steer.log_prob(b["steer"]) + level.log_prob(b["level"].clamp(max=SPEEDS.index(cap))))
        copy = (copy * b["good"]).sum() / b["good"].sum().clamp(min=1)
        logp, entropy, returns, value, _ = imagine(world, actor, critic, states.reshape(-1, states.shape[-1]))
        spread = (torch.quantile(returns, 0.95) - torch.quantile(returns, 0.05)).clamp(min=1)
        advantage = ((returns - value.detach()) / spread).detach()
        improve = -(logp * advantage).mean() - 3e-3 * entropy.mean()
        actor_opt.zero_grad()
        (copy + (improve if step > warmup else 0)).backward()
        actor_opt.step()
        critic_loss = F.mse_loss(value, returns.detach())
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        if step % 250 == 0 or step == policy_steps:
            log.write(f"policy,{step},,,,{copy.item():.4f},{returns.mean().item():.3f}\n")
            print(f"policy {step:5d}  copy-the-driver loss {copy.item():.3f}  imagined return {returns.mean().item():.2f}"
                  + ("  (copying only)" if step <= warmup else ""), flush=True)

    print(evaluate(world, actor, critic, held))
    torch.save(
        {"world": world.state_dict(), "actor": actor.state_dict(), "critic": critic.state_dict(),
         "mean": mean, "std": std, "cap": cap, "speeds": SPEEDS},
        folder / "dreamer.pt",
    )
    export(folder, world, actor, mean, std)
    print(f"saved {folder / 'dreamer.pt'} and, for the Pi, {folder / 'dreamer.onnx'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(int(x) for x in a[:3]), *a[3:4])
