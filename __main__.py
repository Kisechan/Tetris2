import sys
import random
from copy import deepcopy

# 游戏常量定义
MAPWIDTH = 10      # 地图宽度
MAPHEIGHT = 20     # 地图高度

# 玩家颜色标识
currBotColor = 0    # 当前玩家颜色 (0红1蓝)
enemyColor = 1      # 对手颜色

# 游戏状态存储
gridInfo = [
    [[0]*(MAPWIDTH + 2) for _ in range(MAPHEIGHT + 2)],  # 玩家0的地图
    [[0]*(MAPWIDTH + 2) for _ in range(MAPHEIGHT + 2)]   # 玩家1的地图
]

# 转移行相关
trans = [
    [[0]*(MAPWIDTH + 2) for _ in range(6)],  # 玩家0的转移行
    [[0]*(MAPWIDTH + 2) for _ in range(6)]   # 玩家1的转移行
]
transCount = [0, 0]  # 双方转移行数
maxHeight = [0, 0]   # 双方当前最大高度
elimTotal = [0, 0]   # 双方总消除行数
elimCombo = [0, 0]   # 双方连续消除计数
elimBonus = [0, 1, 3, 5, 7]  # 消除行数对应的奖励分数

# 方块类型统计
typeCountForColor = [
    [0]*7,  # 玩家0收到的各类方块数量
    [0]*7   # 玩家1收到的各类方块数量
]

# 7种方块在4种旋转状态下的形状定义
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

attr = [0.0031716211351593346,-1.0,-0.975700323228232,-0.7890873645716108,-0.9075430752097631,-0.909170572114474]
numHolesWeight = attr[0]      # 空洞数的权重
pileHeightWeight = attr[1]    # 平均堆高的权重
wellHeightWeight = attr[2]    # 井深总和的权重
heightDiffWeight = attr[3]    # 行高差异的权重
linesClearWeight = attr[4]    # 消除行数的权重
edgeExposureWeight = attr[5]  # 边缘暴露的权重

def calculateParameters(color, simulatedGrid):
    """计算当前局面的各项参数"""
    params = {
        'numHoles': 0,          # 空洞数量
        'pileHeight': 0,        # 平均堆高
        'wellHeight': 0,        # 井深总和
        'heightDiff': 0,        # 行高差异
        'linesClear': 0,        # 消除行数
        'edgeExposure': 0,      # 边缘暴露
    }

    # 计算各列高度和最大高度
    columnHeights = [0] * (MAPWIDTH + 2)
    for x in range(1, MAPWIDTH+1):
        for y in range(1, MAPHEIGHT+1):
            if simulatedGrid[y][x]:
                columnHeights[x] = MAPHEIGHT - y + 1
                break

    # 空洞检测
    for x in range(1, MAPWIDTH+1):
        foundBlock = False
        for y in range(MAPHEIGHT, 0, -1):
            if simulatedGrid[y][x]:
                foundBlock = True
            elif foundBlock:
                params['numHoles'] += 1

    # 堆高参数
    params['pileHeight'] = max(columnHeights)

    # 井深计算
    for x in range(2, MAPWIDTH):
        left = columnHeights[x-1]
        right = columnHeights[x+1]
        current = columnHeights[x]
        if current < left and current < right:
            params['wellHeight'] += min(left, right) - current

    # 行高差异
    params['heightDiff'] = sum(abs(columnHeights[x] - columnHeights[x+1])
                                for x in range(1, MAPWIDTH))

    # 边缘暴露（两侧边缘列的高度）
    params['edgeExposure'] = columnHeights[1] + columnHeights[MAPWIDTH]

    return params

def evaluatePosition(params):
    """算分"""
    score = 0
    score += params['linesClear'] * linesClearWeight
    score -= params['numHoles'] * numHolesWeight
    score -= params['pileHeight'] * pileHeightWeight
    score -= params['wellHeight'] * wellHeightWeight
    score -= params['heightDiff'] * heightDiffWeight
    score -= params['edgeExposure'] * edgeExposureWeight
    return score

class Tetris:
    """方块类，表示一个俄罗斯方块"""
    def __init__(self, blockType, color):
        self.blockType = blockType  # 方块类型(0-6)
        self.color = color          # 所属玩家(0或1)
        self.shape = blockShape[blockType]  # 形状定义
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
            if tmpX < 1 or tmpX > MAPWIDTH or tmpY < 1 or tmpY > MAPHEIGHT:
                return False
            if gridInfo[self.color][tmpY][tmpX] != 0:
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
                # print(f"Now x={oldX},y={oldY}")
                if gridInfo[self.color][oldY][oldX] == 2:
                    gridInfo[self.color][oldY][oldX] = 1

        # 将方块的4个格子标记为2(表示新放置的方块)
        for i in range(4):
            x = self.blockX + self.shape[self.orientation][2*i]
            y = self.blockY + self.shape[self.orientation][2*i+1]
            gridInfo[self.color][y][x] = 2
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
                if gridInfo[self.color][by][bx] != 0:
                    return False

            fromO = (fromO + 1) % 4  # 尝试下一个中间状态

        return True

def init():
    """初始化游戏地图，设置边界墙"""
    for i in range(MAPHEIGHT + 2):
        gridInfo[0][i][0] = gridInfo[0][i][MAPWIDTH+1] = -2
        gridInfo[1][i][0] = gridInfo[1][i][MAPWIDTH+1] = -2
    for i in range(MAPWIDTH + 2):
        gridInfo[0][0][i] = gridInfo[0][MAPHEIGHT+1][i] = -2
        gridInfo[1][0][i] = gridInfo[1][MAPHEIGHT+1][i] = -2

def checkDirectDropTo(color, blockType, x, y, o):
    """检查方块是否能从顶部直接落到指定位置"""
    shape = blockShape[blockType][o]
    # 从指定位置向上检查路径是否畅通
    for cy in range(y, MAPHEIGHT + 1):
        for i in range(4):
            dx = x + shape[2*i]
            dy = cy + shape[2*i+1]
            if dy > MAPHEIGHT:
                continue
            if dy < 1 or dx < 1 or dx > MAPWIDTH or gridInfo[color][dy][dx] != 0:
                return False
    return True

def eliminate(color):
    """消除满行并处理转移行"""
    count = 0       # 消除行数
    hasBonus = 0     # 是否有连消奖励
    maxHeight[color] = MAPHEIGHT
    newGrid = [[0]*(MAPWIDTH+2) for _ in range(MAPHEIGHT+2)]
    fullRows = []    # 记录满行的y坐标

    # 找出所有满行
    for y in range(1, MAPHEIGHT+1):
        full = all(gridInfo[color][y][x] != 0 for x in range(1, MAPWIDTH+1))
        empty = all(gridInfo[color][y][x] == 0 for x in range(1, MAPWIDTH+1))
        if full:
            fullRows.append(y)
        elif empty:
            maxHeight[color] = y-1
            break

    # 处理满行，生成转移行
    firstFull = True
    for y in fullRows:
        # 连消奖励(连续3回合有消除)
        if firstFull and elimCombo[color] >= 2:
            # 转移行保留边界和原有方块
            trans[color][count] = [
                gridInfo[color][y][x] if gridInfo[color][y][x] in (1, -2) else 0
                for x in range(MAPWIDTH+2)
            ]
            count += 1
            hasBonus = 1
        firstFull = False

        # 将满行加入转移行(去除最后放置的方块)
        trans[color][count] = [
            gridInfo[color][y][x] if gridInfo[color][y][x] in (1, -2) else 0
            for x in range(MAPWIDTH+2)
        ]
        count += 1

    # if count != 0:
    #     print("Full Rows:")
    #     print(fullRows)
    #     for i in fullRows:
    #         print(i, gridInfo[color][i])
    #     print("\nCurrent Grid:")
    #     printCurrentGrid(color)


    transCount[color] = count
    # 更新连消计数
    elimCombo[color] = count // 2 if count > 0 else 0
    # 更新总分数
    elimTotal[color] += elimBonus[count - hasBonus] if count - hasBonus < 5 else 7

    if count == 0:
        return

    # 重新构建地图(消除行后下落)
    writeRow = MAPHEIGHT
    newGrid = [[-2, [0]*MAPWIDTH, -2]]
    for y in range(1, MAPHEIGHT):
        # 跳过被消除的行
        if y not in fullRows:
            # 复制整行数据，保留边界
            newGrid.append(gridInfo[color][y][:])
            writeRow -= 1

    # 顶部填充空行
    while(len(newGrid) <= MAPHEIGHT):
        newGrid.append([-2] + [0]*MAPWIDTH + [-2])

    # 更新地图和边界
    gridInfo[color] = newGrid
    # if len(fullRows) != 0:
    #     print("\nNew Gird:")
    #     for i in range(MAPHEIGHT, 0, -1):
    #         print(i, newGrid[i])
    #     print()

    # 更新最大高度（当前最高方块位置）
    maxHeight[color] = MAPHEIGHT - writeRow

def transfer():
    """处理双方的行转移"""
    color1, color2 = 0, 1
    if transCount[0] == 0 and transCount[1] == 0:
        return -1  # 双方都没有转移行

    # 只有一方有转移行的情况
    if transCount[0] == 0 or transCount[1] == 0:
        if transCount[0] == 0:
            color1, color2 = 1, 0  # 交换双方

        h2 = maxHeight[color2] + transCount[color1]
        if h2 > MAPHEIGHT:
            return color2  # 对方场地溢出，对方输

        # 将对方的行上移
        for y in range(h2, transCount[color1], -1):
            gridInfo[color2][y] = gridInfo[color2][y - transCount[color1]][:]

        # 在底部添加转移行
        for i in range(transCount[color1]):
            gridInfo[color2][i+1] = trans[color1][i][:]

        return -1

    # 双方都有转移行的情况
    else:
        h1 = maxHeight[color1] + transCount[color2]
        h2 = maxHeight[color2] + transCount[color1]

        # 检查是否溢出
        if h1 > MAPHEIGHT: return color1
        if h2 > MAPHEIGHT: return color2

        # 处理color2的场地(添加color1的转移行)
        temp = []
        for y in range(1, transCount[color1]+1):
            temp.append(trans[color1][y-1][:])
        for y in range(transCount[color1]+1, h2+1):
            temp.append(gridInfo[color2][y - transCount[color1]][:])
        gridInfo[color2] = [[-2]*(MAPWIDTH+2)] + temp + [[-2]*(MAPWIDTH+2)]*(MAPHEIGHT - h2)

        # 处理color1的场地(添加color2的转移行)
        temp = []
        for y in range(1, transCount[color2]+1):
            temp.append(trans[color2][y-1][:])
        for y in range(transCount[color2]+1, h1+1):
            temp.append(gridInfo[color1][y - transCount[color2]][:])
        gridInfo[color1] = [[-2]*(MAPWIDTH+2)] + temp + [[-2]*(MAPWIDTH+2)]*(MAPHEIGHT - h1)

        return -1

def canPut(color, blockType):
    """检查是否还能放置指定类型的方块"""
    for y in range(MAPHEIGHT, 0, -1):
        for x in range(1, MAPWIDTH+1):
            for o in range(4):
                t = Tetris(blockType, color).set(x, y, o)
                if t.isValid() and checkDirectDropTo(color, blockType, x, y, o):
                    return True
    return False

def simulatePlace(color, blockX, blockY, rotation, blockType):
    simulated = deepcopy(gridInfo[color])
    for i in range(4):
        x = blockX + blockShape[blockType][rotation][2*i]
        y = blockY + blockShape[blockType][rotation][2*i+1]
        simulated[y][x] = 2
    return simulated

def findBestSpot(color, blockType):
    """寻找最佳落点(改进后的策略)"""
    bestScore = -float('inf')
    bestX, bestY, bestO = 1, 1, 0

    # 遍历所有可能的位置和旋转状态
    for o in range(4):
        for x in range(1, MAPWIDTH+1):
            # 找到能落下的最低位置
            for y in range(1, MAPHEIGHT+1):
                t = Tetris(blockType, color).set(x, y, o)
                if t.isValid() and checkDirectDropTo(color, blockType, x, y, o) and t.onGround():
                    simulated = simulatePlace(color, x, y, o, blockType)
                    params = calculateParameters(color, simulated)
                    score = evaluatePosition(params)
                    if score > bestScore:
                        bestScore = score
                        bestX, bestY, bestO = x, y, o

    return bestX, bestY, bestO

def printCurrentGrid(player):
    for i in range(MAPHEIGHT, 0, -1):
        print(i, gridInfo[player][i])

def main():
    random.seed()
    init()

    # 读取输入
    lines = sys.stdin.read().split('\n')

    ptr = 0

    # 读取回合数
    turnID = int(lines[ptr])
    ptr +=1

    # 第一回合特殊处理
    first = list(map(int, lines[ptr].split()))
    ptr +=1
    blockType, currBotColor = first[0], first[1]
    enemyColor = 1 - currBotColor
    nextTypeForColor = [blockType, blockType]
    typeCountForColor[0][blockType] +=1
    typeCountForColor[1][blockType] +=1

    # print(f"i={0}")
    # printCurrentGrid(1)
    # print()

    # 处理历史回合
    for i in range(1, turnID):
        currType = [nextTypeForColor[0], nextTypeForColor[1]]

        # 读取我方上一步操作
        myAct = list(map(int, lines[ptr].split()))
        ptr +=1
        bt, x, y, o = myAct
        myBlock = Tetris(currType[currBotColor], currBotColor).set(x, y, o)
        myBlock.place()
        typeCountForColor[enemyColor][bt] +=1
        nextTypeForColor[enemyColor] = bt

        # 读取对方上一步操作
        enemyAct = list(map(int, lines[ptr].split()))
        ptr +=1
        bt, x, y, o = enemyAct
        enemyBlock = Tetris(currType[enemyColor], enemyColor).set(x, y, o)
        enemyBlock.place()
        typeCountForColor[currBotColor][bt] +=1
        nextTypeForColor[currBotColor] = bt

        # 消除行和转移处理
        eliminate(0)
        eliminate(1)
        transfer()

        # print(f"i={i},Enemy:")
        # printCurrentGrid(0)
        # print()

    # print("\nNow Enemy:")
    # printCurrentGrid(0)
    #
    # print("\nNow Mine:")
    # printCurrentGrid(1)

    # 决策

    # 当前回合决策
    block = Tetris(nextTypeForColor[currBotColor], currBotColor)

    # 选择落点
    finalX, finalY, finalO = findBestSpot(currBotColor, block.blockType)

    # 选择给对方的方块类型
    maxCount = max(typeCountForColor[enemyColor])
    minCount = min(typeCountForColor[enemyColor])

    if maxCount - minCount == 2:
        # 如果某种方块太多，选择非最大数量的方块
        for bt in range(7):
            if typeCountForColor[enemyColor][bt] != maxCount:
                blockForEnemy = bt
                break
    else:
        # 随机选择，但避免连续给同一种方块
        lastGiven = nextTypeForColor[enemyColor]
        blockForEnemy = random.choice([i for i in range(7) if i != lastGiven or random.random() < 0.3])

    # 输出决策(方块类型, x, y, 旋转状态)
    print(f"{blockForEnemy} {finalX} {finalY} {finalO}")



if __name__ == "__main__":
    main()