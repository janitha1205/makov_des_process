import numpy as np

# States and Actions


def policy_evaluation(theta, gamma, policy, P, V, states, actions):
    while True:
        delta = 0
        for s in range(len(states)):
            v = 0
            print(s)
            V1 = V[s]
            a = policy[s]

            for i in range(len(actions)):
                prob, next_s, reward = P[s][i]
                v += prob * (reward + gamma * next_s)

            V[s] = v
            delta = max(delta, abs(V1 - v))
            print(delta)

        if delta < theta:

            return V


def policy_improvement(policy, states, gamma, P, actions):
    policy_stable = True
    for s in range(len(states)):
        old_action = policy[s]
        action_l = []
        action_values = np.zeros(len(actions))
        for a in range(len(actions)):
            [prob, next_s, reward] = P[s][a]
            action_values[a] = prob * (reward + gamma * next_s)

        best_action = np.argmax(action_values)

        policy[s] = best_action
        if old_action != best_action:

            policy_stable = False
    return policy, policy_stable


# Policy Iteration loop
def mdp(states, actions, P):
    gamma = 0.09
    theta = 0.87

    # Initial policy: all actions = 0
    policy = np.zeros(len(states), dtype=int)
    V = np.zeros(len(states), dtype=int)
    while True:
        V = policy_evaluation(theta, gamma, policy, P, V, states, actions)

        policy, ok = policy_improvement(policy, states, gamma, P, actions)
        if ok:
            break

    print("Optimal Value Function:")
    print(V)
    print("Optimal Policy:")
    print(policy)
    return V, policy
