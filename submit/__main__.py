import torch
import sys
from copy import deepcopy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from collections import deque
import numpy as np
class Tetris2_env:
    def __init__(self):
        # 游戏常量定义
        self.MAPWIDTH = 10      # 地图宽度
        self.MAPHEIGHT = 20     # 地图高度

        # 玩家颜色标识
        self.currBotColor = 0    # 当前玩家颜色 (0红1蓝)
        self.enemyColor = 1      # 对手颜色

        # 游戏状态存储
        self.gridInfo = [
            [[0]*(self.MAPWIDTH + 2) for _ in range(self.MAPHEIGHT + 2)],  # 玩家0的地图
            [[0]*(self.MAPWIDTH + 2) for _ in range(self.MAPHEIGHT + 2)]   # 玩家1的地图
        ]

        # 转移行相关
        self.trans = [
            [[0]*(self.MAPWIDTH + 2) for _ in range(6)],  # 玩家0的转移行
            [[0]*(self.MAPWIDTH + 2) for _ in range(6)]   # 玩家1的转移行
        ]
        self.transCount = [0, 0]  # 双方转移行数
        self.maxHeight = [0, 0]   # 双方当前最大高度
        self.elimTotal = [0, 0]   # 双方总消除行数
        self.elimCombo = [0, 0]   # 双方连续消除计数
        self.elimBonus = [0, 1, 3, 5, 7]  # 消除行数对应的奖励分数

        # 方块类型统计
        self.typeCountForColor = [
            [0]*7,  # 玩家0收到的各类方块数量
            [0]*7   # 玩家1收到的各类方块数量
        ]

        # 7种方块在4种旋转状态下的形状定义
        self.blockShape = [
            # 长L型(0)
            [[0,0,1,0,-1,0,-1,-1], [0,0,0,1,0,-1,1,-1], [0,0,-1,0,1,0,1,1], [0,0,0,-1,0,1,-1,1]],
            # 短L型(1)
            [[0,0,-1,0,1,0,1,-1], [0,0,0,-1,0,1,1,1], [0,0,1,0,-1,0,-1,1], [0,0,0,1,0,-1,-1,-1]],
            # 反Z型(2)
            [[0,0,1,0,0,-1,-1,-1], [0,0,0,1,1,0,1,-1], [0,0,-1,0,0,1,1,1], [0,0,0,-1,-1,0,-1,1]],
            # 正Z型(3)
            [[0,0,-1,0,0,-1,1,-1], [0,0,0,-1,1,0,1,1], [0,0,1,0,0,1,-1,1], [0,0,0,1,-1,0,-1,-1]],
            # T型(4)
            [[0,0,-1,0,0,1,1,0], [0,0,0,-1,-1,0,0,1], [0,0,1,0,0,-1,-1,0], [0,0,0,1,1,0,0,-1]],
            # 长条型(5)
            [[0,0,0,-1,0,1,0,2], [0,0,1,0,-1,0,-2,0], [0,0,0,1,0,-1,0,-2], [0,0,-1,0,1,0,2,0]],
            # 田字型(6)
            [[0,0,0,1,-1,0,-1,1], [0,0,-1,0,0,-1,-1,-1], [0,0,0,-1,1,0,1,-1], [0,0,1,0,0,1,1,1]]
        ]

        # 旋转时需要检查的空格位置
        self.rotateBlank = [
            # 长L型
            [[1,1,0,0], [-1,1,0,0], [-1,-1,0,0], [1,-1,0,0]],
            # 短L型
            [[-1,-1,0,0], [1,-1,0,0], [1,1,0,0], [-1,1,0,0]],
            # 反Z型
            [[1,1,0,0], [-1,1,0,0], [-1,-1,0,0], [1,-1,0,0]],
            # 正Z型
            [[-1,-1,0,0], [1,-1,0,0], [1,1,0,0], [-1,1,0,0]],
            # T型
            [[-1,-1,-1,1,1,1,0,0], [-1,-1,-1,1,1,-1,0,0], [-1,-1,1,1,1,-1,0,0], [-1,1,1,1,1,-1,0,0]],
            # 长条型
            [[1,-1,-1,1,-2,1,-1,2,-2,2], [1,1,-1,-1,-2,-1,-1,-2,-2,-2], [-1,1,1,-1,2,-1,1,-2,2,-2], [-1,-1,1,1,2,1,1,2,2,2]],
            # 田字型
            [[0,0], [0,0], [0,0], [0,0]]
        ]

        for i in range(self.MAPHEIGHT + 2):
            self.gridInfo[0][i][0] = self.gridInfo[0][i][self.MAPWIDTH+1] = -2
            self.gridInfo[1][i][0] = self.gridInfo[1][i][self.MAPWIDTH+1] = -2
        for i in range(self.MAPWIDTH + 2):
            self.gridInfo[0][0][i] = self.gridInfo[0][self.MAPHEIGHT+1][i] = -2
            self.gridInfo[1][0][i] = self.gridInfo[1][self.MAPHEIGHT+1][i] = -2

        # 当前方块和对手方块
        self.current_piece = None
        self.enemy_piece = None
        self.reset()

    def reset(self):
        # 玩家颜色标识
        self.currBotColor = 0    # 当前玩家颜色 (0红1蓝)
        self.enemyColor = 1      # 对手颜色

        # 游戏状态存储
        self.gridInfo = [
            [[0]*(self.MAPWIDTH + 2) for _ in range(self.MAPHEIGHT + 2)],  # 玩家0的地图
            [[0]*(self.MAPWIDTH + 2) for _ in range(self.MAPHEIGHT + 2)]   # 玩家1的地图
        ]

        # 转移行相关
        self.trans = [
            [[0]*(self.MAPWIDTH + 2) for _ in range(6)],  # 玩家0的转移行
            [[0]*(self.MAPWIDTH + 2) for _ in range(6)]   # 玩家1的转移行
        ]
        self.transCount = [0, 0]  # 双方转移行数
        self.maxHeight = [0, 0]   # 双方当前最大高度
        self.elimTotal = [0, 0]   # 双方总消除行数
        self.elimCombo = [0, 0]   # 双方连续消除计数
        self.elimBonus = [0, 1, 3, 5, 7]  # 消除行数对应的奖励分数

        # 方块类型统计
        self.typeCountForColor = [
            [0]*7,  # 玩家0收到的各类方块数量
            [0]*7   # 玩家1收到的各类方块数量
        ]

        for i in range(self.MAPHEIGHT + 2):
            self.gridInfo[0][i][0] = self.gridInfo[0][i][self.MAPWIDTH+1] = -2
            self.gridInfo[1][i][0] = self.gridInfo[1][i][self.MAPWIDTH+1] = -2
        for i in range(self.MAPWIDTH + 2):
            self.gridInfo[0][0][i] = self.gridInfo[0][self.MAPHEIGHT+1][i] = -2
            self.gridInfo[1][0][i] = self.gridInfo[1][self.MAPHEIGHT+1][i] = -2

        # 初始化双方方块
        self.enemy_piece = self.current_piece = np.random.randint(0, 7)

        # 更新方块计数
        self.typeCountForColor[0][self.current_piece] += 1
        self.typeCountForColor[1][self.enemy_piece] += 1
        return self._get_state()

    def checkDirectDropTo(self, color, blockType, x, y, o):
        """检查方块是否能从顶部直接落到指定位置"""
        shape = self.blockShape[blockType][o]
        # 从指定位置向上检查路径是否畅通
        for cy in range(y, self.MAPHEIGHT + 1):
            for i in range(4):
                dx = x + shape[2*i]
                dy = cy + shape[2*i+1]
                if dy > self.MAPHEIGHT:
                    continue
                if dy < 1 or dx < 1 or dx > self.MAPWIDTH or self.gridInfo[color][dy][dx] != 0:
                    return False
        return True

    def eliminate(self, color):
        """消除满行并处理转移行"""
        count = 0       # 消除行数
        hasBonus = 0     # 是否有连消奖励
        self.maxHeight[color] = self.MAPHEIGHT
        newGrid = [[0]*(self.MAPWIDTH+2) for _ in range(self.MAPHEIGHT+2)]
        fullRows = []    # 记录满行的y坐标

        # 找出所有满行
        for y in range(1, self.MAPHEIGHT+1):
            full = all(self.gridInfo[color][y][x] != 0 for x in range(1, self.MAPWIDTH+1))
            empty = all(self.gridInfo[color][y][x] == 0 for x in range(1, self.MAPWIDTH+1))
            if full:
                fullRows.append(y)
            elif empty:
                self.maxHeight[color] = y-1
                break

        # 处理满行，生成转移行
        firstFull = True
        for y in fullRows:
            # 连消奖励(连续3回合有消除)
            if firstFull and self.elimCombo[color] >= 2:
                # 转移行保留边界和原有方块
                self.trans[color][count] = [
                    self.gridInfo[color][y][x] if self.gridInfo[color][y][x] in (1, -2) else 0
                    for x in range(self.MAPWIDTH+2)
                ]
                count += 1
                hasBonus = 1
            firstFull = False

            # 将满行加入转移行(去除最后放置的方块)
            self.trans[color][count] = [
                self.gridInfo[color][y][x] if self.gridInfo[color][y][x] in (1, -2) else 0
                for x in range(self.MAPWIDTH+2)
            ]
            count += 1

        self.transCount[color] = count
        # 更新连消计数
        self.elimCombo[color] = count // 2 if count > 0 else 0
        # 更新总分数
        self.elimTotal[color] += self.elimBonus[count - hasBonus] if count - hasBonus < 5 else 7

        if count == 0:
            return 0

        # 重新构建地图(消除行后下落)
        writeRow = self.MAPHEIGHT
        newGrid = [[-2, [0]*self.MAPWIDTH, -2]]
        for y in range(1, self.MAPHEIGHT):
            # 跳过被消除的行
            if y not in fullRows:
                # 复制整行数据，保留边界
                newGrid.append(self.gridInfo[color][y][:])
                writeRow -= 1

        # 顶部填充空行
        while(len(newGrid) <= self.MAPHEIGHT):
            newGrid.append([-2] + [0]*self.MAPWIDTH + [-2])

        # 更新地图和边界
        self.gridInfo[color] = newGrid

        # 更新最大高度（当前最高方块位置）
        self.maxHeight[color] = self.MAPHEIGHT - writeRow

        return len(fullRows)

    def transfer(self):
        """处理双方的行转移"""
        color1, color2 = 0, 1
        if self.transCount[0] == 0 and self.transCount[1] == 0:
            return -1  # 双方都没有转移行

        # 只有一方有转移行的情况
        if self.transCount[0] == 0 or self.transCount[1] == 0:
            if self.transCount[0] == 0:
                color1, color2 = 1, 0  # 交换双方

            h2 = self.maxHeight[color2] + self.transCount[color1]
            if h2 > self.MAPHEIGHT:
                return color2  # 对方场地溢出，对方输

            # 将对方的行上移
            for y in range(h2, self.transCount[color1], -1):
                self.gridInfo[color2][y] = self.gridInfo[color2][y - self.transCount[color1]][:]

            # 在底部添加转移行
            for i in range(self.transCount[color1]):
                self.gridInfo[color2][i+1] = self.trans[color1][i][:]

            return -1

        # 双方都有转移行的情况
        else:
            h1 = self.maxHeight[color1] + self.transCount[color2]
            h2 = self.maxHeight[color2] + self.transCount[color1]

            # 检查是否溢出
            if h1 > self.MAPHEIGHT: return color1
            if h2 > self.MAPHEIGHT: return color2

            # 处理color2的场地(添加color1的转移行)
            temp = []
            for y in range(1, self.transCount[color1]+1):
                temp.append(self.trans[color1][y-1][:])
            for y in range(self.transCount[color1]+1, h2+1):
                temp.append(self.gridInfo[color2][y - self.transCount[color1]][:])
            self.gridInfo[color2] = [[-2]*(self.MAPWIDTH+2)] + temp + [[-2]*(self.MAPWIDTH+2)]*(self.MAPHEIGHT - h2)

            # 处理color1的场地(添加color2的转移行)
            temp = []
            for y in range(1, self.transCount[color2]+1):
                temp.append(self.trans[color2][y-1][:])
            for y in range(self.transCount[color2]+1, h1+1):
                temp.append(self.gridInfo[color1][y - self.transCount[color2]][:])
            self.gridInfo[color1] = [[-2]*(self.MAPWIDTH+2)] + temp + [[-2]*(self.MAPWIDTH+2)]*(self.MAPHEIGHT - h1)

            return -1

    def canPut(self, color, blockType):
        """检查是否还能放置指定类型的方块"""
        for y in range(self.MAPHEIGHT, 0, -1):
            for x in range(1, self.MAPWIDTH+1):
                for o in range(4):
                    t = Tetris(blockType, color, self).set(x, y, o)
                    if t.isValid() and self.checkDirectDropTo(color, blockType, x, y, o):
                        return True
        return False

    def simulatePlace(self, color, blockX, blockY, rotation, blockType):
        simulated = deepcopy(self.gridInfo[color])
        for i in range(4):
            x = blockX + self.blockShape[blockType][rotation][2*i]
            y = blockY + self.blockShape[blockType][rotation][2*i+1]
            simulated[y][x] = 2
        return simulated

    def render(self, mode='human'):
        """可视化游戏状态"""
        print(f"Current Player: {self.currBotColor}")
        print("Current Piece:", self.current_piece)
        print("Enemy Piece:", self.enemy_piece)

        # 定义字符表示
        SYMBOLS = {
            -2: '|',  # 墙
            0: ' ',   # 空气
            1: 'O',   # 旧方块
            2: 'X',   # 新方块
        }

        def print_grid(player_id):
            print(f"\nPlayer {player_id} Grid:")
            # 打印列号（横坐标）
            print("   ", end="")
            for j in range(1, self.MAPWIDTH + 1):
                print(f"{j%10}", end="")
            print()

            # 打印每一行（从上往下）
            for i in range(self.MAPHEIGHT, 0, -1):
                print(f"{i:2d} ", end="")  # 打印行号
                for j in range(1, self.MAPWIDTH + 1):
                    cell = self.gridInfo[player_id][i][j]
                    print(SYMBOLS.get(cell, '?'), end="")
                print()

        # 分别打印两个玩家的地图
        print_grid(0)
        print("="*30)
        print_grid(1)
        print("="*30)

    def get_valid_actions(self):
        valid_actions = []
        # 检查棋盘状态，返回所有合法动作（位置和旋转的组合）
        for o in range(4):
            for x in range(1, self.MAPWIDTH + 1):
                # 找到能落下的最低位置
                for y in range(1, self.MAPHEIGHT + 1):
                    t = Tetris(self.current_piece, self.currBotColor, self).set(x, y, o)
                    if t.isValid() and self.checkDirectDropTo(self.currBotColor, self.current_piece, x, y, o) and t.onGround():
                        valid_actions.append(encode_action(x, y, o, self.MAPWIDTH))
                        # print(f"Valid Action: x {x} y {y} o {o}")
        return valid_actions

    def step(self, action):
        """
        执行一步动作
        action: 包含两个部分 (placement_action, next_block_action)
        placement_action: (x, y, o) 当前方块的放置位置和旋转
        next_block_action: 为对手选择的方块类型 (0-6)
        """
        placement_action, next_block_action = action

        # 1. 检查动作合法性
        if not self._is_valid_action(placement_action, next_block_action):
            return self._get_state(), -1, True, {"reason": "invalid action"}  # 非法动作直接判负

        # 2. 放置当前方块
        x, y, o = placement_action
        tetris = Tetris(self.current_piece, self.currBotColor, self).set(x, y, o)
        if not tetris.isValid():
            return self._get_state(), -1, True, {"reason": "invalid placement"}

        # print(f"x = {x}, y = {y}, o = {o}")
        # 放置方块到地图
        self.gridInfo = tetris.place()
        if self.gridInfo is None:
            print("gridInfo is None")
            return self._get_state(), -1, True, {"reason": "invalid placement"}

        # self.render()
        # 3. 消除行并处理转移行
        reward = self.eliminate(self.currBotColor) * 1.0
        result = self.transfer()

        # 4. 检查游戏是否结束
        if result != -1:
            winner = result
            reward = 10.0 if winner == self.currBotColor else -10.0
            return self._get_state(), reward, True, {"winner": winner}

        # 5. 为对手选择方块 (需满足极差<=2的条件)
        if not self._is_valid_next_block(next_block_action):
            return self._get_state(), -1, True, {"reason": "invalid next block"}

        # 更新方块计数
        self.typeCountForColor[self.enemyColor][next_block_action] += 1

        # 6. 交换玩家并更新方块
        self.currBotColor, self.enemyColor = self.enemyColor, self.currBotColor
        self.current_piece, self.enemy_piece = self.enemy_piece, next_block_action

        # 7. 检查新玩家是否能放置方块
        if not self.canPut(self.currBotColor, self.current_piece):
            winner = self.enemyColor  # 当前玩家无法放置，对手获胜
            reward = 10.0 if winner == 0 else -10.0  # 假设我们总是训练玩家0
            return self._get_state(), reward, True, {"winner": winner}

        # 8. 计算中间奖励
        reward += self._calculate_intermediate_reward()

        return self._get_state(), reward, False, {}

    def _is_valid_action(self, placement_action, next_block_action):
        """检查动作是否合法"""
        # 检查放置动作
        x, y, rotation = placement_action
        if not (1 <= x <= self.MAPWIDTH and 1 <= y <= self.MAPHEIGHT and 0 <= rotation <= 3):
            return False

        # 检查方块类型
        if not (0 <= next_block_action <= 6):
            return False

        # 检查是否能直接下落
        if not self.checkDirectDropTo(self.currBotColor, self.current_piece, x, y, rotation):
            return False

        return True

    def _is_valid_next_block(self, block_type):
        """检查选择的方块是否满足极差<=2的条件"""
        counts = self.typeCountForColor[self.enemyColor]
        new_counts = counts.copy()
        new_counts[block_type] += 1
        return max(new_counts) - min(new_counts) <= 2

    def _calculate_intermediate_reward(self):
        """计算中间奖励"""
        # 存活奖励
        reward = 0.05

        # 1. 消除行奖励
        # reward += self.elimTotal[self.currBotColor] * 1.0

        # 2. 高度惩罚 (鼓励保持低高度)
        reward -= self.maxHeight[self.currBotColor] * 0.01

        # 3. 孔洞惩罚 (鼓励减少孔洞)
        holes = self._count_holes(self.currBotColor)
        reward -= holes * 0.01

        return reward

    def _count_holes(self, color):
        """计算棋盘上的孔洞数量"""
        holes = 0
        for x in range(1, self.MAPWIDTH+1):
            found_block = False
            for y in range(self.MAPHEIGHT, 0, -1):
                if self.gridInfo[color][y][x] != 0:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def _get_state(self):
        """获取当前游戏状态"""
        # 将双方棋盘和当前方块信息转换为神经网络输入
        state = {
            "current_grid": np.array([row[1:self.MAPWIDTH+1] for row in self.gridInfo[self.currBotColor][1:self.MAPHEIGHT+1]]),
            "enemy_grid": np.array([row[1:self.MAPWIDTH+1] for row in self.gridInfo[self.enemyColor][1:self.MAPHEIGHT+1]]),
            "current_piece": self.current_piece,
            "enemy_piece": self.enemy_piece,
            "piece_counts": np.array(self.typeCountForColor[self.currBotColor]),
            "enemy_piece_counts": np.array(self.typeCountForColor[self.enemyColor]),
            "max_height": self.maxHeight[self.currBotColor],
            "enemy_max_height": self.maxHeight[self.enemyColor]
        }
        return state

class Tetris:
    """方块类，表示一个俄罗斯方块"""
    def __init__(self, blockType, color, Tetris2_env):
        self.env = Tetris2_env
        self.blockType = blockType  # 方块类型(0-6)
        self.color = color          # 所属玩家(0或1)
        self.shape = self.env.blockShape[blockType]  # 形状定义
        self.blockX = -1            # 当前x坐标(未初始化)
        self.blockY = -1            # 当前y坐标(未初始化)
        self.orientation = 0        # 当前旋转状态(0-3)

    def set(self, x=-1, y=-1, o=-1):
        """设置方块的位置和旋转状态"""
        if x != -1: self.blockX = x
        if y != -1: self.blockY = y
        if o != -1: self.orientation = o
        return self  # 返回自身以支持链式调用

    def isValid(self, x=-1, y=-1, o=-1):
        """检查当前位置和旋转状态是否有效"""
        x = x if x != -1 else self.blockX
        y = y if y != -1 else self.blockY
        o = o if o != -1 else self.orientation

        # 检查旋转状态是否合法
        if o < 0 or o > 3:
            return False

        # 检查方块的4个格子是否都在地图内且为空
        for i in range(4):
            tmpX = x + self.shape[o][2*i]
            tmpY = y + self.shape[o][2*i+1]
            if tmpX < 1 or tmpX > self.env.MAPWIDTH or tmpY < 1 or tmpY > self.env.MAPHEIGHT:
                return False
            if self.env.gridInfo[self.color][tmpY][tmpX] != 0:
                return False
        return True

    def onGround(self):
        """检查方块是否已经落地(不能再下落)"""
        return self.isValid() and not self.isValid(-1, self.blockY-1)

    def place(self):
        """将方块放置到地图上"""
        if not self.onGround():
            return None

        # 将旧方块标记为1
        for oldY in range(1, self.env.MAPHEIGHT):
            for oldX in range(1, self.env.MAPWIDTH+1):
                if self.env.gridInfo[self.color][oldY][oldX] == 2:
                    self.env.gridInfo[self.color][oldY][oldX] = 1

        # 将方块的4个格子标记为2(表示新放置的方块)
        for i in range(4):
            x = self.blockX + self.shape[self.orientation][2*i]
            y = self.blockY + self.shape[self.orientation][2*i+1]
            self.env.gridInfo[self.color][y][x] = 2
        return self.env.gridInfo

    def rotation(self, o):
        """检查能否旋转到指定状态"""
        if o < 0 or o > 3:
            return False
        if self.orientation == o:
            return True

        fromO = self.orientation
        while True:
            # 检查中间状态是否有效
            if not self.isValid(-1, -1, fromO):
                return False
            if fromO == o:
                break

            # 检查旋转过程中需要为空的格子
            blank = self.env.rotateBlank[self.blockType][fromO]
            for i in range(0, len(blank), 2):
                if i >= len(blank): break
                bx = self.blockX + blank[i]
                by = self.blockY + blank[i+1]
                if bx == self.blockX and by == self.blockY:
                    break
                if self.env.gridInfo[self.color][by][bx] != 0:
                    return False

            fromO = (fromO + 1) % 4  # 尝试下一个中间状态

        return True

def encode_action(x, y, o, width):
    # 注意这里 x 和 y 是从 1 开始的（与环境中一致）
    return (y - 1) * width * 4 + (x - 1) * 4 + o

def decode_action(idx, width):
    o = idx % 4
    x = ((idx // 4) % width) + 1
    y = ((idx // (4 * width)) % 100) + 1  # 100 是最大高度，避免无限扩展
    return x, y, o

class PPONetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PPONetwork, self).__init__()

        # 棋盘编码器 (处理两个棋盘)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # 计算卷积后的尺寸
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2*padding - kernel_size) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1])))
        linear_input_size = convw * convh * 64

        # 方块类型和统计信息编码
        self.piece_embedding = nn.Embedding(7, 16)  # 7种方块类型
        self.stat_fc = nn.Linear(14 + 2, 64)  # 14个计数+2个高度

        # 合并所有特征
        self.combine_fc = nn.Linear(linear_input_size * 2 + 16 + 64, 512)

        # 输出头
        self.placement_head = nn.Linear(512, input_shape[1] * input_shape[2] * 4)  # 每个位置和旋转的概率
        self.next_block_head = nn.Linear(512, 7)  # 选择下一个方块
        self.value_head = nn.Linear(512, 1)  # 状态价值

    def forward(self, state):
        # 处理当前棋盘
        current_grid = state["current_grid"].unsqueeze(1).float()  # [batch, 1, H, W]
        c1 = torch.relu(self.conv1(current_grid))
        c2 = torch.relu(self.conv2(c1))
        c3 = torch.relu(self.conv3(c2))
        current_features = c3.view(c3.size(0), -1)

        # 处理对手棋盘
        enemy_grid = state["enemy_grid"].unsqueeze(1).float()
        e1 = torch.relu(self.conv1(enemy_grid))
        e2 = torch.relu(self.conv2(e1))
        e3 = torch.relu(self.conv3(e2))
        enemy_features = e3.view(e3.size(0), -1)

        # 处理当前方块
        piece_emb = self.piece_embedding(state["current_piece"].long())

        # 处理统计信息
        stats = torch.cat([
            state["piece_counts"].float(),
            state["enemy_piece_counts"].float(),
            state["max_height"].unsqueeze(1).float(),
            state["enemy_max_height"].unsqueeze(1).float()
        ], dim=1)
        stats_features = torch.relu(self.stat_fc(stats))

        # 合并所有特征
        combined = torch.cat([current_features, enemy_features, piece_emb, stats_features], dim=1)
        combined = torch.relu(self.combine_fc(combined))

        # 计算输出
        placement_logits = self.placement_head(combined)
        next_block_logits = self.next_block_head(combined)
        value = self.value_head(combined)

        return placement_logits, next_block_logits, value

class PPOAgent:
    def __init__(self, input_shape, num_actions, lr=3e-4, gamma=0.99, clip_epsilon=0.2, beta=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.memory = deque(maxlen=10000)

    def get_action(self, state, env):
        state = self._preprocess_state(state)

        width = state["current_grid"].shape[-1]
        height = state["current_grid"].shape[-2]
        rotation_size = 4
        total_actions = width * height * rotation_size

        valid_actions = env.get_valid_actions()  # 是一维索引列表

        with torch.no_grad():
            placement_logits, next_block_logits, value = self.policy(state)
            placement_logits = placement_logits.view(-1)

            placement_mask = torch.full_like(placement_logits, float('-inf'))
            if valid_actions:
                for idx in valid_actions:
                    placement_mask[idx] = placement_logits[idx]
            else:
                print("没有合法的放置动作，将随机选择")
                placement_mask = placement_logits

            placement_probs = torch.softmax(placement_mask, dim=0)
            placement_dist = Categorical(placement_probs)
            placement_action_flat = placement_dist.sample()
            x, y, o = decode_action(placement_action_flat.item(), width)
            placement_action = (x, y, o)

            legal_next_blocks = []
            piece_count = env.typeCountForColor
            opponent_color = 1 - env.currBotColor

            for next_block in range(7):
                # 假设选择了该方块，更新对方的方块数量
                temp_piece_count = piece_count[opponent_color][:]
                temp_piece_count[next_block] += 1

                # 检查对方方块数量的极差是否超过2
                if max(temp_piece_count) - min(temp_piece_count) <= 2:
                    legal_next_blocks.append(next_block)

            next_block_logits = next_block_logits.view(-1)
            next_block_mask = torch.full_like(next_block_logits, float('-inf'))
            if legal_next_blocks:
                for idx in legal_next_blocks:
                    next_block_mask[idx] = next_block_logits[idx]
            else:
                print("无合法选择方块动作，将随机选择")
                next_block_mask = next_block_logits

            next_block_probs = torch.softmax(next_block_mask, dim=-1)
            next_block_dist = Categorical(next_block_probs)
            next_block_action = next_block_dist.sample()

            # 计算动作概率
            action_prob = placement_dist.log_prob(placement_action_flat).exp() * \
                          next_block_dist.log_prob(next_block_action).exp()

            return (placement_action, next_block_action.item()), action_prob.item(), value.item()


    def store_transition(self, state, action, prob, value, reward, done):
        self.memory.append((state, action, prob, value, reward, done))

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        states, actions, old_probs, old_values, rewards, dones = zip(*samples)

        # 预处理所有状态
        states = [self._preprocess_state(s) for s in states]
        batched_states = self._batch_states(states)

        # 折扣回报和优势
        returns = []
        advantages = []
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
            advantages.insert(0, R - old_values[i])

        advantages = torch.tensor(advantages, device=self.device).float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.tensor(returns, device=self.device).float()
        old_probs = torch.tensor(old_probs, device=self.device).float()
        old_values = torch.tensor(old_values, device=self.device).float()

        # 获取动作张量
        width = batched_states["current_grid"].shape[-1]
        height = batched_states["current_grid"].shape[-2]

        placement_actions = []
        next_block_actions = []
        for action in actions:
            placement, next_block = action
            placement_flat = (placement[1]-1)*width*4 + (placement[0]-1)*4 + placement[2]
            placement_actions.append(placement_flat)
            next_block_actions.append(next_block)

        placement_actions = torch.tensor(placement_actions, device=self.device).long()
        next_block_actions = torch.tensor(next_block_actions, device=self.device).long()

        # 预测
        placement_logits, next_block_logits, values = self.policy(batched_states)

        placement_probs = torch.softmax(placement_logits.view(batch_size, -1), dim=-1)
        placement_dist = Categorical(placement_probs)
        placement_log_probs = placement_dist.log_prob(placement_actions)

        next_block_probs = torch.softmax(next_block_logits, dim=-1)
        next_block_dist = Categorical(next_block_probs)
        next_block_log_probs = next_block_dist.log_prob(next_block_actions)

        new_probs = (placement_log_probs + next_block_log_probs).exp()
        ratios = new_probs / old_probs

        # PPO目标函数
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)

        # 熵奖励
        entropy = placement_dist.entropy().mean() + next_block_dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss - self.beta * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _preprocess_state(self, state):
        """将单个状态 dict 转换为张量格式"""
        processed = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                processed[k] = torch.FloatTensor(v).unsqueeze(0).to(self.device)
            else:
                processed[k] = torch.tensor([v], device=self.device)
        return processed

    def _batch_states(self, state_list):
        """将多个预处理后的状态 dict 合并为 batched dict"""
        batch = {}
        for key in state_list[0]:
            batch[key] = torch.cat([s[key] for s in state_list], dim=0)
        return batch

def botzone_keeprunning():
    print("\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
    sys.stdout.flush()

def main():
    env = Tetris2_env()
    input_shape = (1, env.MAPHEIGHT, env.MAPWIDTH)
    agent = PPOAgent(input_shape, num_actions=7)

    agent.policy.load_state_dict(torch.load("data/tetris_ppo_500.pth", map_location=torch.device('cpu')))
    agent.policy.eval()

    turn = int(input())
    t, c = input().split()
    t = int(t)
    c = int(c)

    env.current_piece = t
    env.enemy_piece = t
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
    botzone_keeprunning()

    while(True):
        line = input().split()
        if len(line) != 4:
            continue
        t, x, y, o = line
        t = int(t)
        x = int(x)
        y = int(y)
        o = int(o)

        env.current_piece = t

        tetris = Tetris(env.enemy_piece, env.enemyColor, env).set(x, y, o)
        env.gridInfo = tetris.place()
        # env.render()
        env.eliminate(env.enemyColor)
        env.transfer()

        state = env._get_state()
        action, _, _ = agent.get_action(state, env)

        env.enemy_piece = next_block

        placement, next_block = action
        finalX, finalY, finalO = placement

        tetris = Tetris(env.current_piece, env.currBotColor, env).set(finalX, finalY, finalO)
        env.gridInfo = tetris.place()
        env.eliminate(env.currBotColor)
        env.transfer()

        print(f"{next_block} {finalX} {finalY} {finalO}")
        botzone_keeprunning()

if __name__ == "__main__":
    main()