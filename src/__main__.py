import torch
import random
from copy import deepcopy
from ppo.tetris2_env import Tetris2_env, Tetris
from ppo.train import PPOAgent
def main():
    env = Tetris2_env()
    input_shape = (1, env.MAPHEIGHT, env.MAPWIDTH)
    agent = PPOAgent(input_shape, num_actions=7)

    agent.policy.load_state_dict(torch.load("ppo/model/tetris_ppo_500.pth"))
    agent.policy.eval()

    t, c = input().split()
    t = int(t)
    c = int(c)

    env.current_piece = t
    env.currBotColor = c
    env.enemyColor = 1 - c

    state = env._get_state()
    action, _, _ = agent.get_action(state, env)

    placement, next_block = action
    finalX, finalY, finalO = placement

    tetris = Tetris(env.current_piece, env.currBotColor, env).set(finalX, finalY, finalO)
    env.gridInfo = tetris.place()
    env.eliminate(env.currBotColor)
    env.transfer()

    print(f"{next_block} {finalX} {finalY} {finalO}")

    while(True):
        t, x, y, o = input().split()
        t = int(t)
        x = int(x)
        y = int(y)
        o = int(o)

        env.current_piece = t

        tetris = Tetris(next_block, env.enemyColor, env).set(x, y, o)
        env.gridInfo = tetris.place()
        env.eliminate(env.enemyColor)
        env.transfer()

        state = env._get_state()
        action, _, _ = agent.get_action(state, env)

        placement, next_block = action
        finalX, finalY, finalO = placement

        tetris = Tetris(env.current_piece, env.currBotColor, env).set(finalX, finalY, finalO)
        env.gridInfo = tetris.place()
        env.eliminate(env.currBotColor)
        env.transfer()

        print(f"{next_block} {finalX} {finalY} {finalO}")

if __name__ == "__main__":
    main()