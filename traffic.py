import random
import numpy as np


class TrafficEnv:
    """Simple 4-way junction: action 0 = North-South green, 1 = East-West green."""

    def __init__(self, max_q=10, steps=100, seed=None):
        self.rng = random.Random(seed)
        self.max_q, self.steps = max_q, steps
        self.reset()

    def reset(self):
        self.ns, self.ew = self.rng.randint(0, 5), self.rng.randint(0, 5)
        self.green, self.t = 0, 0
        return self._state()

    def _level(self, q):          # 0..3  (low ... very high)
        return min(q // 3, 3)

    def _state(self):
        return (self._level(self.ns), self._level(self.ew), self.green)

    def step(self, action):
        switch_penalty = 1 if action != self.green else 0
        self.green = action
        self.ns = min(self.max_q, self.ns + self.rng.choice([0, 1, 1, 2]))   # arrivals
        self.ew = min(self.max_q, self.ew + self.rng.choice([0, 0, 1, 2]))
        if action == 0:                                                      # departures
            self.ns = max(0, self.ns - 3)
        else:
            self.ew = max(0, self.ew - 3)
        self.t += 1
        reward = -(self.ns + self.ew) - switch_penalty
        return self._state(), reward, self.t >= self.steps


def train(episodes=2000, alpha=0.1, gamma=0.9, eps=1.0, eps_min=0.05, decay=0.995, seed=0):
    env = TrafficEnv(seed=seed)
    Q = np.zeros((4, 4, 2, 2))                     # Q[ns, ew, green, action]
    rewards = []
    for ep in range(episodes):
        s, done, total = env.reset(), False, 0
        while not done:
            if random.random() < eps:
                a = random.randint(0, 1)           # explore
            else:
                a = int(np.argmax(Q[s]))           # exploit
            s2, r, done = env.step(a)
            Q[s][a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s][a])   # Q-learning update
            s, total = s2, total + r
        eps = max(eps_min, eps * decay)
        rewards.append(total)
    return Q, rewards


def evaluate(policy, episodes=200, seed=123):
    env = TrafficEnv(seed=seed)
    scores = []
    for _ in range(episodes):
        s, done, total, t = env.reset(), False, 0, 0
        while not done:
            s, r, done = env.step(policy(s, t))
            total, t = total + r, t + 1
        scores.append(total)
    return float(np.mean(scores))


if __name__ == "__main__":
    random.seed(0)
    Q, rewards = train()
    print("RL (Q-learning):", evaluate(lambda s, t: int(np.argmax(Q[s]))))
    print("Fixed timer    :", evaluate(lambda s, t: (t // 5) % 2))
    print("Random         :", evaluate(lambda s, t: random.randint(0, 1)))