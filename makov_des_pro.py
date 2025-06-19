import numpy as np

# States and Actions
states = [0, 1, 2, 3]
actions = [0, 1, 2]

# Transition model P[s][a] = [(prob, next_state, reward)]
P = {
    0: {0: [(1.0, 0, 3)], 1: [(1.0, 1, 2)], 2: [(1.0, 0, 10)]},
    1: {0: [(1.0, 0, 0)], 1: [(1.0, 2, 1)], 2: [(1.0, 0, 10)]},
    2: {0: [(1.0, 1, 3)], 1: [(1.0, 2, 0)], 2: [(1.0, 0, 10)]},
    3: {0: [(1.0, 1, 3)], 1: [(1.0, 2, 0)], 2: [(1.0, 0, 10)]},
}

gamma = 0.9
theta = 1e-6

# Initial policy: all actions = 0
policy = np.zeros(len(states), dtype=int)
V = np.zeros(len(states))


def policy_evaluation(policy):
    while True:
        delta = 0
        for s in states:
            v = V[s]
            a = policy[s]
            V[s] = sum(
                prob * (reward + gamma * V[next_state])
                for prob, next_state, reward in P[s][a]
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break


def policy_improvement():
    policy_stable = True
    for s in states:
        old_action = policy[s]
        action_values = np.zeros(len(actions))
        for a in actions:
            action_values[a] = sum(
                prob * (reward + gamma * V[next_state])
                for prob, next_state, reward in P[s][a]
            )
        best_action = np.argmax(action_values)
        policy[s] = best_action
        if old_action != best_action:
            policy_stable = False
    return policy_stable


# Policy Iteration loop
while True:
    policy_evaluation(policy)
    if policy_improvement():
        break

print("Optimal Value Function:")
print(V)
print("Optimal Policy:")
print(policy)
