import sys
import random
import math
import copy
from copy import deepcopy

MAPWIDTH = 10
MAPHEIGHT = 20
attr = [-0.13, -0.80, -0.18, -0.04, 0.27]

blockShape = [
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
rotateBlank = [
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

class TetrisGame:
    def __init__(self):
        self.currBotColor = 0
        self.enemyColor = 1
        self.moves_made = 0
        self.gridInfo = [
            [[0]*(MAPWIDTH + 2) for _ in range(MAPHEIGHT + 2)],  # 玩家0的地图
            [[0]*(MAPWIDTH + 2) for _ in range(MAPHEIGHT + 2)]   # 玩家1的地图
        ]
        self.trans = [
            [[0]*(MAPWIDTH + 2) for _ in range(6)],  # 玩家0的转移行
            [[0]*(MAPWIDTH + 2) for _ in range(6)]   # 玩家1的转移行
        ]
        self.transCount = [0, 0]  # 双方转移行数
        self.maxHeight = [0, 0]   # 双方当前最大高度
        self.elimTotal = [0, 0]   # 双方总消除行数
        self.elimCombo = [0, 0]   # 双方连续消除计数
        self.elimBonus = [0, 1, 3, 5, 7]  # 消除行数对应的奖励分数
        self.currBlockType = [0, 0]
        self.nextBlockType = [0, 0]

        # 方块类型统计
        self.typeCountForColor = [
            [0]*7,  # 玩家0收到的各类方块数量
            [0]*7   # 玩家1收到的各类方块数量
        ]

        self.occupiedWeight = attr[0]
        self.numHolesWeight = attr[1]
        self.pileHeightWeight = attr[2]
        self.wellHeightWeight = attr[3]
        self.linesClearWeight = attr[4]

        self.color = None           # 当前回合
        self.blockType = None       # 方块类型(0-6)
        self.shape = None           # 形状定义
        self.blockX = -1            # 当前x坐标(未初始化)
        self.blockY = -1            # 当前y坐标(未初始化)
        self.orientation = -1       # 当前旋转状态(0-3)
        self.lose = -1


    def calculateParameters(self, simulatedGrid):
        """计算当前局面的各项参数"""
        params = {
            'occupied': 0,
            'numHoles': 0,
            'pileHeight': 0,
            'wellHeight': 0,
            'linesClear': 0,
        }

        for y in range(1, MAPHEIGHT + 1):
            for x in range(1, MAPWIDTH + 1):
                if simulatedGrid[y][x] != 0:
                    params['occupied'] += 1

        for x in range(1, MAPWIDTH + 1):
            y = MAPHEIGHT
            while y > 0 and simulatedGrid[y][x] == 0:
                y -= 1
            y -= 1
            while y > 0:
                if simulatedGrid[y][x] == 0:
                    params['numHoles'] += 1
                y -= 1

        for y in range(MAPHEIGHT, 0, -1):
            found = False
            for x in range(1, MAPWIDTH + 1):
                if simulatedGrid[y][x] != 0:
                    params['pileHeight'] = y
                    found = True
                    break
            if found:
                break

        pre = -1
        for x in range(MAPWIDTH, 0, -1):
            ht = 0
            for y in range(MAPHEIGHT, 0, -1):
                if simulatedGrid[y][x] != 0:
                    ht = y
                    break
            if pre != -1:
                params['wellHeight'] += abs(ht - pre)
            pre = ht

        for y in range(1, MAPHEIGHT + 1):
            ct = 0
            for x in range(1, MAPWIDTH + 1):
                if simulatedGrid[y][x] != 0:
                    ct += 1
            if ct == MAPWIDTH:
                params['linesClear'] += 1

        return params

    def evaluatePosition(self, params):
        """算分"""
        score = 0
        score += params['occupied'] * self.occupiedWeight
        score += params['numHoles'] * self.numHolesWeight
        score += params['pileHeight'] * self.pileHeightWeight
        score += params['wellHeight'] * self.wellHeightWeight
        score += params['linesClear'] * self.linesClearWeight
        return score

    def init(self):
        """初始化游戏地图，设置边界墙"""
        for i in range(MAPHEIGHT + 2):
            self.gridInfo[0][i][0] = self.gridInfo[0][i][MAPWIDTH+1] = -2
            self.gridInfo[1][i][0] = self.gridInfo[1][i][MAPWIDTH+1] = -2
        for i in range(MAPWIDTH + 2):
            self.gridInfo[0][0][i] = self.gridInfo[0][MAPHEIGHT+1][i] = -2
            self.gridInfo[1][0][i] = self.gridInfo[1][MAPHEIGHT+1][i] = -2
        for i in range(1, MAPWIDTH + 1):
            for j in range(1, MAPHEIGHT + 1):
                self.gridInfo[0][j][i] = 0
                self.gridInfo[1][j][i] = 0

    def set(self, x=-1, y=-1, o=-1):
        """设置方块的位置和旋转状态"""
        if x != -1: self.blockX = x
        if y != -1: self.blockY = y
        if o != -1: self.orientation = o

    def setBlock(self, Type):
        self.blockType = Type

    def setColor(self, Color):
        self.color = Color

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
            tmpX = x + blockShape[self.blockType][o][2*i]
            tmpY = y + blockShape[self.blockType][o][2*i+1]
            if tmpX < 1 or tmpX > MAPWIDTH or tmpY < 1 or tmpY > MAPHEIGHT:
                return False
            if self.gridInfo[self.color][tmpY][tmpX] != 0:
                return False
        return True

    def onGround(self):
        """检查方块是否已经落地(不能再下落)"""
        return self.isValid() and not self.isValid(-1, self.blockY-1)

    def place(self):
        """将方块放置到地图上"""
        if not self.onGround():
            return False

        # 将旧方块标记为1
        for oldY in range(1, MAPHEIGHT):
            for oldX in range(1, MAPWIDTH+1):
                if self.gridInfo[self.color][oldY][oldX] == 2:
                    self.gridInfo[self.color][oldY][oldX] = 1

        # 将方块的4个格子标记为2(表示新放置的方块)
        for i in range(4):
            x = self.blockX + blockShape[self.blockType][self.orientation][2*i]
            y = self.blockY + blockShape[self.blockType][self.orientation][2*i+1]
            self.gridInfo[self.color][y][x] = 2
        return True

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
            blank = rotateBlank[self.blockType][fromO]
            for i in range(0, len(blank), 2):
                if i >= len(blank): break
                bx = self.blockX + blank[i]
                by = self.blockY + blank[i+1]
                if bx == self.blockX and by == self.blockY:
                    break
                if self.gridInfo[self.color][by][bx] != 0:
                    return False

            fromO = (fromO + 1) % 4  # 尝试下一个中间状态

        return True

    def checkDirectDropTo(self, x, y, o):
        """检查方块是否能从顶部直接落到指定位置"""
        shape = blockShape[self.blockType][o]
        # 从指定位置向上检查路径是否畅通
        for cy in range(y, MAPHEIGHT + 1):
            for i in range(4):
                dx = x + shape[2*i]
                dy = cy + shape[2*i+1]
                if dy > MAPHEIGHT:
                    continue
                if dy < 1 or dx < 1 or dx > MAPWIDTH or self.gridInfo[self.color][dy][dx] != 0:
                    return False
        return True

    def eliminate(self, color):
        """消除满行并处理转移行"""
        count = 0       # 消除行数
        hasBonus = 0     # 是否有连消奖励
        self.maxHeight[color] = MAPHEIGHT
        fullRows = []    # 记录满行的y坐标
        fillRows = []

        # 找出所有满行
        for y in range(1, MAPHEIGHT+1):
            full = all(self.gridInfo[color][y][x] != 0 for x in range(1, MAPWIDTH+1))
            empty = all(self.gridInfo[color][y][x] == 0 for x in range(1, MAPWIDTH+1))
            if full:
                fullRows.append(y)
            elif empty:
                self.maxHeight[color] = y-1
                break
            else:
                fillRows.append(y)

        # 处理满行，生成转移行
        firstFull = True
        for y in fullRows:
            # 连消奖励(连续3回合有消除) 连消疑似乱写
            if firstFull and self.elimCombo[color] >= 2:
                # 转移行保留边界和原有方块
                self.trans[color][count] = [
                    self.gridInfo[color][y][x] if self.gridInfo[color][y][x] in (1, -2) else 0
                    for x in range(MAPWIDTH+2)
                ]
                count += 1
                hasBonus = 1
            firstFull = False

            # 将满行加入转移行(去除最后放置的方块)
            self.trans[color][count] = [
                self.gridInfo[color][y][x] if self.gridInfo[color][y][x] in (1, -2) else 0
                for x in range(MAPWIDTH+2)
            ]
            count += 1

        if firstFull == False:
            self.elimCombo[color] += 1
        else:
            self.elimCombo[color] = 0

        self.transCount[color] = count
        # 更新总分数
        self.elimTotal[color] += self.elimBonus[count - hasBonus] if count - hasBonus < 5 else 7

        if count == 0:
            return

        # 重新构建地图(消除行后下落)
        newGrid = [[-2] + [-2]*MAPWIDTH + [-2]]
        for y in range(1, MAPHEIGHT + 1):
            # 跳过被消除的行
            if y in fillRows:
                # 复制整行数据，保留边界
                newGrid.append(self.gridInfo[color][y][:])
            elif y not in fullRows:
                break

        self.maxHeight[color] = len(newGrid) - 1

        # 顶部填充空行
        while (len(newGrid) <= MAPHEIGHT):
            newGrid.append([-2] + [0]*MAPWIDTH + [-2])

        newGrid.append([-2] + [-2]*MAPWIDTH + [-2])
        # 更新地图和边界
        self.gridInfo[color] = newGrid


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
            if h2 > MAPHEIGHT:
                self.lose = color2
                return

            # 将对方的行上移
            for y in range(h2, self.transCount[color1], -1):
                self.gridInfo[color2][y] = self.gridInfo[color2][y - self.transCount[color1]][:]

            # 在底部添加转移行
            for i in range(self.transCount[color1]):
                self.gridInfo[color2][i+1] = self.trans[color1][i][:]

            return -1

        # 双方都有转移行的情况
        else:
            h1 = self.maxHeight[color1] + self.transCount[color2] - self.transCount[color1]
            h2 = self.maxHeight[color2] + self.transCount[color1] - self.transCount[color2]

            # 检查是否溢出
            if h1 > MAPHEIGHT:
                self.lose = color1
                return
            if h2 > MAPHEIGHT:
                self.lose = color2
                return

            # 处理color2的场地(添加color1的转移行)
            temp1 = []
            temp2 = []
            for y in range(self.transCount[color1]):
                temp1.append(self.trans[color1][y][:])
            for y in range(1, self.maxHeight[color2] + 1):
                temp1.append(self.gridInfo[color2][y][:])
            while len(temp1) < MAPHEIGHT:
                temp1.append([-2] + [0]*MAPWIDTH + [-2])

            # 处理color1的场地(添加color2的转移行)
            for y in range(self.transCount[color2]):
                temp2.append(self.trans[color2][y][:])
            for y in range(1, self.maxHeight[color1] + 1):
                temp2.append(self.gridInfo[color1][y][:])
            while len(temp2) < MAPHEIGHT:
                temp2.append([-2] + [0]*MAPWIDTH + [-2])

            self.gridInfo[color2] = [[-2]*(MAPWIDTH+2)] + temp1 + [[-2]*(MAPWIDTH+2)]
            self.gridInfo[color1] = [[-2]*(MAPWIDTH+2)] + temp2 + [[-2]*(MAPWIDTH+2)]

            return -1

    def canPut(self):
        """检查是否还能放置指定类型的方块"""
        for y in range(MAPHEIGHT, 0, -1):
            for x in range(1, MAPWIDTH+1):
                for o in range(4):
                    self.set(x, y, o)
                    if self.isValid() and self.checkDirectDropTo(x, y, o):
                        return True
        return False

    def simulatePlace(self, placeX, placeY, rotation, placeType):
        simulated = deepcopy(self.gridInfo[self.color])
        for i in range(4):
            x = placeX + blockShape[placeType][rotation][2*i]
            y = placeY + blockShape[placeType][rotation][2*i+1]
            simulated[y][x] = 2
        return simulated

    def findBestSpot(self):
        """寻找最佳落点(改进后的策略)"""
        bestScore = -float('inf')
        bestX, bestY, bestO = 1, 1, 0

        # 遍历所有可能的位置和旋转状态
        for o in range(4):
            for x in range(1, MAPWIDTH+1):
                # 找到能落下的最低位置
                for y in range(1, MAPHEIGHT+1):
                    self.set(x, y, o)
                    if not self.isValid() or not self.onGround():
                        continue
                    if self.checkDirectDropTo(x, y, o):
                        simulated = self.simulatePlace(x, y, o, self.blockType)
                        params = self.calculateParameters(simulated)
                        score = self.evaluatePosition(params)
                        if score > bestScore:
                            bestScore = score
                            bestX, bestY, bestO = x, y, o

        return bestX, bestY, bestO

    def println(self, color):
        for j in range(MAPHEIGHT + 1, -1, -1):
            for i in range(0, MAPWIDTH + 2):
                print(self.gridInfo[color][j][i], end=" ")
            print()

    def calWorst(self, blockType, color):
        self.blockType = blockType
        self.color = color

        bestScore = -float('inf')

        # 遍历所有可能的位置和旋转状态
        for o in range(4):
            for x in range(1, MAPWIDTH+1):
                # 找到能落下的最低位置
                for y in range(1, MAPHEIGHT+1):
                    self.set(x, y, o)
                    if self.isValid() and self.checkDirectDropTo(x, y, o) and self.onGround():
                        simulated = self.simulatePlace(x, y, o, self.blockType)
                        params = self.calculateParameters(simulated)
                        score = self.evaluatePosition(params)
                        bestScore = max(bestScore, score)

        return bestScore

    """
    def get_legal_moves(self):
        legal = []
        legalType = []
        minCount = min(self.typeCountForColor[1 - self.color])
        for i in range(7):
            if self.typeCountForColor[1 - self.color][i] < minCount + 2:
                legalType.append(i)
        for o in range(4):
            for x in range(1, MAPWIDTH+1):
                for y in range(1, MAPHEIGHT+1):
                    self.set(x, y, o)
                    if self.isValid() and self.checkDirectDropTo(x, y, o) and self.onGround():
                        for q in legalType:
                            legal.append((x, y, o, q))
        if len(legal) == 0:
            self.lose = self.color
        return legal
    """

    def get_legal_moves(self, K = 6):
        # 1) 先算出所有 (x,y,o) 的合法落点
        candidates = []
        legal = []
        for o in range(4):
            for x in range(1, MAPWIDTH+1):
                for y in range(1, MAPHEIGHT+1):
                    self.set(x, y, o)
                    if not self.isValid() or not self.onGround():
                        continue
                    if self.checkDirectDropTo(x, y, o):
                        sim = self.simulatePlace(x, y, o, self.blockType)
                        score = self.evaluatePosition(self.calculateParameters(sim))
                        candidates.append(((x, y, o), score))
        # 2) 按 score 排序，取前 K 个
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_xyz = [move for move,_ in candidates[:K]]

        legalType = []
        minCount = min(self.typeCountForColor[1 - self.color])
        for i in range(7):
            if self.typeCountForColor[1 - self.color][i] < minCount + 2:
                legalType.append(i)

        for (x,y,o) in top_xyz:
            for b in legalType:
                legal.append((x, y, o, b))
        return legal

    def apply_move(self, move):
        x, y, o, TypeForEnemy = move
        self.set(x, y, o)
        self.place()

    def evaluate(self):
        """
        对当前局面进行评估，返回己方分值与对手分值之差。
        假设 self.color 表示当前玩家，对手为 1 - self.color。
        """
        my_params = self.calculateParameters(self.gridInfo[self.color])
        opp_params = self.calculateParameters(self.gridInfo[1 - self.color])
        my_score = self.evaluatePosition(my_params)
        opp_score = self.evaluatePosition(opp_params)
        return my_score - opp_score

    def updateBlock(self):
        self.currBlockType = copy.deepcopy(self.nextBlockType)
        self.nextBlockType = [0, 0]

    def advance_turn(self):
        """
        更新回合信息：当双方均已行动后执行行消除和行转移，
        然后重置计数器，为下一回合做准备，并切换当前决策方。
        """
        self.moves_made += 1
        if self.moves_made >= 2:
            self.eliminate(0)
            self.eliminate(1)
            self.transfer()
            # 重置计数器，表示新一轮开始
            self.moves_made = 0
            self.updateBlock()
            # 如果需要在模型中切换决策方（比如 MCTS 中 current_player），也在这里更新：
        self.color = 1 - self.color
        self.setBlock(self.currBlockType[self.color])

EXPLORATION_CONSTANT = 1.41  # UCB 中的探索常数

class MCTSNode:
    def __init__(self, state, move=None, parent=None, player=0):
        """
        state: 游戏状态（需是 TetrisGame 的一个深拷贝，并实现上面提到的接口）
        move: 从父状态到该状态的落点动作，通常形式为 (x, y, o)
        parent: 父节点
        player: 该节点对应的走子玩家标记（例如 0 或 1）
        """
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.legalMove = state.get_legal_moves()
        self.visits = 0
        self.total_value = 0
        self.player = player

    def is_fully_expanded(self):
        """判断当前节点是否已扩展所有合法动作"""
        return len(self.children) == len(self.legalMove)

    def best_child(self, exploration=EXPLORATION_CONSTANT):
        """
        选择当前节点下的最佳子节点。
        利用 UCB1 公式： score = (child.total_value/child.visits) + exploration * sqrt( ln(parent.visits)/child.visits )
        当 exploration 参数为 0 时，即只基于利用值选最佳动作。
        """
        best_score = -float('inf')
        best_node = None
        for child in self.children:
            exploit = child.total_value / child.visits if child.visits > 0 else 0
            explore = math.sqrt(math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
            score = exploit + exploration * explore
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def expand(self):
        """
        扩展一个未尝试过的动作，返回新扩展的子节点。
        这里随机选择一个尚未扩展的合法动作进行扩展。
        """
        tried_moves = [child.move for child in self.children]
        untried_moves = [m for m in self.legalMove if m not in tried_moves]
        if not untried_moves:
            return None
        move = random.choice(untried_moves)
        new_state = deepcopy(self.state)
        new_state.apply_move(move)
        new_state.advance_turn()  # 切换到对手或下一个回合
        child_node = MCTSNode(new_state, move=move, parent=self, player=new_state.color)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, value):
        """
        将一次模拟的评估结果反向传播至整棵树上。
        value：本次模拟返回的评估值（从根节点视角）
        """
        self.visits += 1
        self.total_value += value
        if self.parent:
            self.parent.backpropagate(value)


def rollout_policy(state):
    """
    简单的 rollout 策略：在状态中随机选择一个合法动作。
    你也可以在这里加入一定的启发式改进。
    """
    legal_moves = state.get_legal_moves()
    if not legal_moves:
        return None
    return random.choice(legal_moves)


def rollout(state, rollout_depth = 3):
    """
    从给定状态开始展开随机模拟（rollout）直至达到 rollout_depth 或者无法行动，
    并返回最终状态的评估值。评估函数应以根节点玩家的角度给出结果。
    """
    current_state = deepcopy(state)
    current_depth = 0
    current_color = current_state.color
    while current_depth < rollout_depth:
        if current_state.lose != -1:
            if current_state.lose == current_color:
                return -float('inf')
            else:
                return float('inf')
        move = rollout_policy(current_state)
        current_state.apply_move(move)
        current_state.advance_turn()
        current_depth += 1
        if current_state.lose != -1:
            if current_state.lose == current_color:
                return -float('inf')
            else:
                return float('inf')

    final_evaluation = current_state.evaluate()
    if current_state.color != current_color:
        final_evaluation *= -1
    return current_state.evaluate()

def mcts_search(root_state, iterations=100, rollout_depth=3):
    """
    对根状态进行蒙特卡洛树搜索，执行若干次迭代后返回最佳动作。
    参数：
      root_state: 当前局面（应为 TetrisGame 状态），需要保证实现上述接口；
      iterations: MCTS 迭代次数；
      rollout_depth: 每次模拟的最大深度（回合数）。
    返回：
      最优的落子动作（例如 (x, y, o)），供当前回合落子使用。
    """
    root_node = MCTSNode(deepcopy(root_state), player=root_state.currBotColor)

    iterations = 20
    for i in range(iterations):
        node = root_node

        # ===== 选择阶段 =====
        # 如果当前节点子节点全部被探索完毕 就选择一个最优子节点
        # 最优子节点的选择是探索和利用加权后的结果
        while node.children and node.is_fully_expanded():
            node = node.best_child()

        # ===== 扩展阶段 =====
        # 遇到子节点没有被完全探索完毕的节点 进行探索
        if not node.is_fully_expanded():
            child = node.expand()
            if child:
                node = child

        # ===== 模拟阶段 =====
        # 随机模拟三个回合 以模拟后的结果作为当前局面的权值
        value = rollout(node.state, rollout_depth)

        # ===== 反向传播阶段 =====
        node.backpropagate(value)

    # 选择访问次数最多或利用值最高的子节点作为最终动作
    best_node = root_node.best_child(exploration=0)
    return best_node.move if best_node is not None else None


def main():
    random.seed()
    tetris = TetrisGame()
    tetris.init()

    # 读取输入
    lines = sys.stdin.read().split('\n')
    ptr = 0
    # 读取回合数
    turnID = int(lines[ptr])
    ptr += 1

    # 第一回合特殊处理
    first = list(map(int, lines[ptr].split()))
    ptr += 1
    blockType, tetris.currBotColor = first[0], first[1]
    tetris.enemyColor = 1 - tetris.currBotColor

    nextTypeForColor = [blockType, blockType]
    tetris.typeCountForColor[0][blockType] += 1
    tetris.typeCountForColor[1][blockType] += 1
    tetris.currBlockType[0] = tetris.currBlockType[1] = blockType

    # 处理历史回合
    for i in range(1, turnID):
        currType = [nextTypeForColor[0], nextTypeForColor[1]]

        # 读取我方上一步操作
        myAct = list(map(int, lines[ptr].split()))
        ptr += 1
        bt, x, y, o = myAct
        tetris.setBlock(currType[tetris.currBotColor])
        tetris.setColor(tetris.currBotColor)
        tetris.set(x, y, o)
        tetris.place()
        tetris.typeCountForColor[tetris.enemyColor][bt] += 1
        nextTypeForColor[tetris.enemyColor] = bt
        tetris.nextBlockType[tetris.enemyColor] = bt

        # 读取敌方上一步操作
        enemyAct = list(map(int, lines[ptr].split()))
        ptr += 1
        bt, x, y, o = enemyAct
        tetris.setBlock(currType[tetris.enemyColor])
        tetris.setColor(tetris.enemyColor)
        tetris.set(x, y, o)
        tetris.place()
        tetris.typeCountForColor[tetris.currBotColor][bt] += 1
        nextTypeForColor[tetris.currBotColor] = bt
        tetris.nextBlockType[tetris.currBotColor] = bt

        # 消除行和转移处理
        tetris.eliminate(0)
        tetris.eliminate(1)
        tetris.transfer()
        tetris.updateBlock()


    # 当前回合决策
    tetris.setColor(tetris.currBotColor)
    tetris.setBlock(nextTypeForColor[tetris.currBotColor])

    # 选择落点
    finalX, finalY, finalO, blockForEnemy = mcts_search(tetris)

    # 输出决策(方块类型, x, y, 旋转状态)
    print(f"{blockForEnemy} {finalX} {finalY} {finalO}")



if __name__ == "__main__":
    main()