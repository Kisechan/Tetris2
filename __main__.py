import sys
import math
import random
import copy
from typing import List, Tuple

# Constants
INF = 1e180
MAPWIDTH = 10
MAPHEIGHT = 20

# Weights for evaluation
W1 = 1.2    # Prevent long bars
W2 = 0.998  # Prefer placing on sides
W3 = 0.98
W4 = 3      # Future considerations
W5 = 1.4    # Capping
W6 = 1
prize = 0
W7 = 30     # More long bars
W8 = 10     # Max height
W9 = 1      # Reject unreachable rows
W10 = 100   # Height coefficient
W11 = 2     # Unreachable
W12 = 100   # Reward for consecutive clears
W13 = 1.4
paper = 5

# Configuration
Work_num = 3  # Number of candidate moves to consider
Type_num = 3  # Number of piece types to consider
M_DEEP = 4    # Search depth for main move
E_DEEP = 2    # Search depth for enemy piece selection
intwo = 1     # Enable special rules

# Piece definitions
GO = [
    [1,1,1,1],  # All rotations allowed for piece 0
    [1,1,1,1],  # Piece 1
    [1,1,0,0],  # Piece 2
    [1,1,0,0],  # Piece 3
    [1,1,1,1],  # Piece 4
    [1,1,0,0],  # Piece 5
    [1,0,0,0]   # Piece 6
]

TYPEwork = [False, False, False, False, True, True, False]

# Piece shapes (7 types, 4 rotations, 4 blocks each)
blockShape = [
    [[0,0,1,0,-1,0,-1,-1], [0,0,0,1,0,-1,1,-1], [0,0,-1,0,1,0,1,1], [0,0,0,-1,0,1,-1,1]],  # L
    [[0,0,-1,0,1,0,1,-1], [0,0,0,-1,0,1,1,1], [0,0,1,0,-1,0,-1,1], [0,0,0,1,0,-1,-1,-1]],  # 反L
    [[0,0,1,0,0,-1,-1,-1], [0,0,0,1,1,0,1,-1], [0,0,-1,0,0,1,1,1], [0,0,0,-1,-1,0,-1,1]],  # Z
    [[0,0,-1,0,0,-1,1,-1], [0,0,0,-1,1,0,1,1], [0,0,1,0,0,1,-1,1], [0,0,0,1,-1,0,-1,-1]],  # 反Z
    [[0,0,-1,0,0,1,1,0], [0,0,0,-1,-1,0,0,1], [0,0,1,0,0,-1,-1,0], [0,0,0,1,1,0,0,-1]],    # T
    [[0,0,0,-1,0,1,0,2], [0,0,1,0,-1,0,-2,0], [0,0,0,1,0,-1,0,-2], [0,0,-1,0,1,0,2,0]],   # I
    [[0,0,0,1,-1,0,-1,1], [0,0,-1,0,0,-1,-1,-1], [0,0,0,-1,1,-0,1,-1], [0,0,1,0,0,1,1,1]]  # 田
]

# 旋转时需要检查的空白位置 [7种类型][4种朝向][最多10个坐标]
rotateBlank = [
    [[1,1,0,0], [-1,1,0,0], [-1,-1,0,0], [1,-1,0,0]],  # L
    [[-1,-1,0,0], [1,-1,0,0], [1,1,0,0], [-1,1,0,0]],  # 反L
    [[1,1,0,0], [-1,1,0,0], [-1,-1,0,0], [1,-1,0,0]],  # Z
    [[-1,-1,0,0], [1,-1,0,0], [1,1,0,0], [-1,1,0,0]],  # 反Z
    [[-1,-1,-1,1,1,1,0,0], [-1,-1,-1,1,1,-1,0,0], [-1,-1,1,1,1,-1,0,0], [-1,1,1,1,1,-1,0,0]],  # T
    [[1,-1,-1,1,-2,1,-1,2,-2,2], [1,1,-1,-1,-2,-1,-1,-2,-2,-2], [-1,1,1,-1,2,-1,1,-2,2,-2], [-1,-1,1,1,2,1,1,2,2,2]],  # I
    [[0,0], [0,0], [0,0], [0,0]]  # 田
]

# Elimination bonuses
elimBonus = [0, 1, 3, 5, 7]

# Game state
currBotColor = 0  # 0: red, 1: blue
enemyColor = 1
gridInfo = [[[0 for _ in range(MAPWIDTH + 2)] for _ in range(MAPHEIGHT + 2)] for _ in range(2)]
trans = [[[0 for _ in range(MAPWIDTH + 2)] for _ in range(6)] for _ in range(2)]
transCount = [0, 0]
maxHeight = [0, 0]
elimTotal = [0, 0]
elimCombo = [0, 0]
typeCountForColor = [[0]*7 for _ in range(2)]
nextTypeForColor = [0, 0]
turnID = 0
blockType = 0
MMH = 0
debug = 0

# Precomputed sums for well calculation
sum_n = [0,1,3,6,10,15,21,28,36,45,55,66,78,91,105,120,136,153,171,190,210]

# Final decision variables
finalX, finalY, finalO = 0, 0, 0
_finalX, _finalY, _finalO = 0, 0, 0

class Tetris:
    def __init__(self, blockType, color):
        self.blockType = blockType
        self.shape = blockShape[blockType]
        self.color = color
        self.blockX = 0
        self.blockY = 0
        self.orientation = 0

    def set(self, x=-1, y=-1, o=-1):
        if x != -1:
            self.blockX = x
        if y != -1:
            self.blockY = y
        if o != -1:
            self.orientation = o
        return self

    def isValid(self, x=-1, y=-1, o=-1):
        x = self.blockX if x == -1 else x
        y = self.blockY if y == -1 else y
        o = self.orientation if o == -1 else o

        if o < 0 or o > 3:
            return False

        for i in range(4):
            tmpX = x + self.shape[o][2 * i]
            tmpY = y + self.shape[o][2 * i + 1]
            if (tmpX < 1 or tmpX > MAPWIDTH or
                    tmpY < 1 or tmpY > MAPHEIGHT or
                    gridInfo[self.color][tmpY][tmpX] != 0):
                return False
        return True

    def onGround(self):
        return self.isValid() and not self.isValid(-1, self.blockY - 1)

    def place(self):
        if not self.onGround():
            return False

        for i in range(4):
            tmpX = self.blockX + self.shape[self.orientation][2 * i]
            tmpY = self.blockY + self.shape[self.orientation][2 * i + 1]
            gridInfo[self.color][tmpY][tmpX] = 2
        return True

    def rotation(self, o):
        if o < 0 or o > 3:
            return False

        if self.orientation == o:
            return True

        fromO = self.orientation
        while True:
            if not self.isValid(-1, -1, fromO):
                return False

            if fromO == o:
                break

            # Check rotation collision
            if intwo:
                for i in range(5):
                    blankX = self.blockX + rotateBlank[self.blockType][fromO][2 * i]
                    blankY = self.blockY + rotateBlank[self.blockType][fromO][2 * i + 1]
                    if blankX == self.blockX and blankY == self.blockY:
                        break
                    if gridInfo[self.color][blankY][blankX] != 0:
                        return False

            fromO = (fromO + 1) % 4
        return True

def init():
    """Initialize the game board with borders"""
    for i in range(MAPHEIGHT + 2):
        gridInfo[1][i][0] = gridInfo[1][i][MAPWIDTH + 1] = -2
        gridInfo[0][i][0] = gridInfo[0][i][MAPWIDTH + 1] = -2
    for i in range(MAPWIDTH + 2):
        gridInfo[1][0][i] = gridInfo[1][MAPHEIGHT + 1][i] = -2
        gridInfo[0][0][i] = gridInfo[0][MAPHEIGHT + 1][i] = -2

def isValid(map, type, x, y, o):
    """Check if a piece position is valid"""
    if o < 0 or o > 3:
        return False

    for i in range(4):
        tmpX = x + blockShape[type][o][2 * i]
        tmpY = y + blockShape[type][o][2 * i + 1]
        if (tmpX < 1 or tmpX > MAPWIDTH or
                tmpY < 1 or tmpY > MAPHEIGHT or
                map[tmpY][tmpX] != 0):
            return False
    return True

def onGround(map, type, x, y, o):
    """Check if a piece is on the ground"""
    return isValid(map, type, x, y, o) and not isValid(map, type, x, y-1, o)

def rotation(map, type, x, y, o):
    """Check if rotation is possible"""
    if intwo:
        for i in range(5):
            blankX = x + rotateBlank[type][o][2 * i]
            blankY = y + rotateBlank[type][o][2 * i + 1]
            if blankX == x and blankY == y:
                break
            if map[blankY][blankX]:
                return False
    return True

def checkDirectDropTo(map, blockType, x, y, o):
    """Check if piece can drop directly to position from top"""
    shape = blockShape[blockType][o]
    while y <= MAPHEIGHT:
        for i in range(4):
            _x = shape[2*i] + x
            _y = shape[2*i+1] + y
            if _y > MAPHEIGHT:
                continue
            if _y < 1 or _x < 1 or _x > MAPWIDTH or map[_y][_x]:
                return False
        y += 1
    return True

def dfs(map, bo, type, x, y, o):
    """Depth-first search to explore all reachable positions"""
    if not isValid(map, type, x, y, o):
        return
    if bo[y][x][o]:
        return

    bo[y][x][o] = True
    dfs(map, bo, type, x-1, y, o)
    dfs(map, bo, type, x+1, y, o)
    dfs(map, bo, type, x, y-1, o)
    if rotation(map, type, x, y, o):
        dfs(map, bo, type, x, y, (o+1)%4)

def initReachability(map, bo, type, de=False):
    """Initialize reachability matrix for a piece type"""
    global debug
    debug = de
    for y in range(MAPHEIGHT + 2):
        for x in range(MAPWIDTH + 2):
            for o in range(4):
                bo[y][x][o] = False

    for o in range(4):
        for x in range(1, MAPWIDTH + 1):
            if checkDirectDropTo(map, type, x, MAPHEIGHT, o):
                dfs(map, bo, type, x, MAPHEIGHT, o)
            if checkDirectDropTo(map, type, x, MAPHEIGHT-1, o):
                dfs(map, bo, type, x, MAPHEIGHT-1, o)
            if type == 5:  # Long bar can drop from higher
                if checkDirectDropTo(map, type, x, MAPHEIGHT-2, o):
                    dfs(map, bo, type, x, MAPHEIGHT-2, o)

def eliminate(color):
    """Eliminate completed rows and prepare transfers"""
    count = transCount[color] = 0
    firstFull = 1
    hasBonus = 0
    maxHeight[color] = MAPHEIGHT

    for i in range(1, MAPHEIGHT + 1):
        emptyFlag = True
        fullFlag = True
        for j in range(1, MAPWIDTH + 1):
            if gridInfo[color][i][j] == 0:
                fullFlag = False
            else:
                emptyFlag = False

        if fullFlag:
            # Check for combo bonus
            if intwo and firstFull and (elimCombo[color] + 1) >= 3:
                for j in range(1, MAPWIDTH + 1):
                    trans[color][count][j] = 1 if gridInfo[color][i][j] == 1 else 0
                count += 1
                hasBonus = 1

            firstFull = 0
            for j in range(1, MAPWIDTH + 1):
                trans[color][count][j] = 1 if gridInfo[color][i][j] == 1 else 0
                gridInfo[color][i][j] = 0
            count += 1
        elif emptyFlag:
            maxHeight[color] = i - 1
            break
        else:
            for j in range(1, MAPWIDTH + 1):
                if gridInfo[color][i][j] > 0:
                    gridInfo[color][i - count + hasBonus][j] = 1
                else:
                    gridInfo[color][i - count + hasBonus][j] = gridInfo[color][i][j]
                if count:
                    gridInfo[color][i][j] = 0

    if count == 0:
        elimCombo[color] = 0
    else:
        elimCombo[color] += 1

    maxHeight[color] -= count - hasBonus
    elimTotal[color] += elimBonus[count]

def transfer():
    """Transfer eliminated rows between players"""
    color1, color2 = 0, 1
    if transCount[color1] == 0 and transCount[color2] == 0:
        return -1

    if transCount[color1] == 0 or transCount[color2] == 0:
        if transCount[color1] == 0 and transCount[color2] > 0:
            color1, color2 = color2, color1

        h2 = maxHeight[color2] + transCount[color1]
        maxHeight[color2] = h2
        if h2 > MAPHEIGHT:
            return color2

        # Move rows up
        for i in range(h2, transCount[color1], -1):
            for j in range(1, MAPWIDTH + 1):
                gridInfo[color2][i][j] = gridInfo[color2][i - transCount[color1]][j]

        # Add transferred rows
        for i in range(transCount[color1], 0, -1):
            for j in range(1, MAPWIDTH + 1):
                gridInfo[color2][i][j] = trans[color1][i - 1][j]
        return -1
    else:
        h1 = maxHeight[color1] + transCount[color2]
        h2 = maxHeight[color2] + transCount[color1]
        maxHeight[color1] = h1
        maxHeight[color2] = h2

        if h1 > MAPHEIGHT:
            return color1
        if h2 > MAPHEIGHT:
            return color2

        # Transfer from color1 to color2
        for i in range(h2, transCount[color1], -1):
            for j in range(1, MAPWIDTH + 1):
                gridInfo[color2][i][j] = gridInfo[color2][i - transCount[color1]][j]

        for i in range(transCount[color1], 0, -1):
            for j in range(1, MAPWIDTH + 1):
                gridInfo[color2][i][j] = trans[color1][i - 1][j]

        # Transfer from color2 to color1
        for i in range(h1, transCount[color2], -1):
            for j in range(1, MAPWIDTH + 1):
                gridInfo[color1][i][j] = gridInfo[color1][i - transCount[color2]][j]

        for i in range(transCount[color2], 0, -1):
            for j in range(1, MAPWIDTH + 1):
                gridInfo[color1][i][j] = trans[color2][i - 1][j]

        return -1

def canPut(color, blockType):
    """Check if a piece can be placed on the board"""
    t = Tetris(blockType, color)
    for y in range(MAPHEIGHT, 0, -1):
        for x in range(1, MAPWIDTH + 1):
            for o in range(4):
                t.set(x, y, o)
                if t.isValid() and checkDirectDropTo(gridInfo[color], blockType, x, y, o):
                    return True
    return False

def GetBoardTransitions(map):
    """Calculate board transitions (changes between filled and empty cells)"""
    mmh = 0

    # Horizontal transitions
    for y in range(1, MAPHEIGHT + 1):
        for x in range(1, MAPWIDTH):
            if (map[y][x] and not map[y][x+1]) or (not map[y][x] and map[y][x+1]):
                mmh += 1

    # Vertical transitions
    for y in range(1, MAPHEIGHT):
        for x in range(1, MAPWIDTH + 1):
            if (map[y][x] and not map[y+1][x]) or (not map[y][x] and map[y+1][x]):
                mmh += 1

    return mmh

def GetBoardBuriedHoles(map):
    """Calculate number of buried holes (empty spaces below blocks)"""
    mmh = 0

    for x in range(1, MAPWIDTH + 1):
        # Find highest block in column
        y = MAPHEIGHT
        while y > 0 and not map[y][x]:
            y -= 1

        # Count holes below highest block
        while y > 0:
            if not map[y][x]:
                mmh += 1
            y -= 1

    return mmh

def GetBoardWells(map):
    """Calculate wells (empty spaces between blocks)"""
    mmh = 0

    for x in range(1, MAPWIDTH + 1):
        wells = 0
        f = False
        W = 1.5

        for y in range(MAPHEIGHT, -1, -1):
            if map[y][x] == 0:
                if map[y][x-1] != 0 and map[y][x+1] != 0:
                    wells += 1
            else:
                mmh += sum_n[wells] * W
                wells = 0
                if f:
                    W += 0.1
                else:
                    W = 0.8
                f = True

    return mmh

def bfs(map):
    """Breadth-first search to find unreachable spaces"""
    qx = []
    qy = []
    l = r = 0
    f = [[0,0,1,-1], [1,-1,0,0]]
    bo = [[False for _ in range(MAPWIDTH + 2)] for _ in range(MAPHEIGHT + 2)]

    # Start from top row
    for x in range(1, MAPWIDTH + 1):
        if not map[MAPHEIGHT][x]:
            qx.append(x)
            qy.append(MAPHEIGHT)
            r += 1

    while l < r:
        for i in range(4):
            nx = qx[l] + f[1][i]
            ny = qy[l] + f[0][i]
            if (1 <= nx <= MAPWIDTH and 1 <= ny <= MAPHEIGHT and
                    not map[ny][nx] and not bo[ny][nx]):
                bo[ny][nx] = True
                qx.append(nx)
                qy.append(ny)
                r += 1
        l += 1

    # Count unreachable empty spaces
    mmh = 0
    for x in range(1, MAPWIDTH + 1):
        for y in range(1, MAPHEIGHT + 1):
            if not map[y][x] and not bo[y][x]:
                mmh += 1

    return mmh

def Mavis(_map, type, x, y, o, deepth, MMH, currBotColor, eRound):
    """Evaluate a potential move"""
    map = copy.deepcopy(_map)
    emptynum = [0] * (MAPHEIGHT + 2)
    sta = [0] * (MAPHEIGHT + 2)
    reach = [True] * (MAPHEIGHT + 2)
    cell_reach = [[False for _ in range(MAPWIDTH + 2)] for _ in range(MAPHEIGHT + 2)]

    # Check piece type balance
    myCount = [False] * 7
    minCount = min(typeCountForColor[currBotColor])
    for i in range(7):
        myCount[i] = (typeCountForColor[currBotColor][i] + 1 - minCount <= 2)

    # Place the piece
    for i in range(4):
        tmpX = x + blockShape[type][o][2*i]
        tmpY = y + blockShape[type][o][2*i+1]
        map[tmpY][tmpX] = 3  # Mark as newly placed

    pop = 0
    erodedShape = 0

    # Check for completed rows
    for _y in range(1, MAPHEIGHT + 1):
        full = True
        for _x in range(1, MAPWIDTH + 1):
            if not map[_y][_x]:
                full = False
                break

        if full:
            pop += 1
            # Count how many of the cleared blocks were from the new piece
            for _x in range(1, MAPWIDTH + 1):
                erodedShape += (map[_y][_x] == 3)
                map[_y][_x] = -1  # Mark for elimination

    # Clear completed rows
    for _x in range(1, MAPWIDTH + 1):
        p = 1
        for i in range(1, MAPHEIGHT + 1):
            if map[i][_x] != -1:
                map[p][_x] = map[i][_x]
                p += 1
        for i in range(p, MAPHEIGHT + 1):
            map[i][_x] = 0

    # Calculate height cost
    P_P = 0
    max_height = 0
    for _x in range(1, MAPWIDTH + 1):
        # Find highest block in column
        _y = MAPHEIGHT
        while _y > 0 and not map[_y][_x]:
            _y -= 1

        # Cost increases with height squared
        cost = _y * _y
        u = x - 5 if x > 5 else x
        cost *= pow(1.05, u)
        P_P += cost
        if _y > max_height:
            max_height = _y

    # Calculate various board metrics
    BoardTransitions = GetBoardTransitions(map)
    BuriedHoles = GetBoardBuriedHoles(map)
    Wells = GetBoardWells(map)
    Bfs = bfs(map)

    # Combine metrics into evaluation score
    holenum = math.sqrt(P_P)/3 + 2.5*max_height + 2*BoardTransitions + 7*BuriedHoles + 2*Bfs + Wells

    # Reward for consecutive clears
    if pop and eRound > 1:
        holenum -= W12

    # If we're not at max depth, consider future moves
    if deepth > 0:
        if deepth == 1:
            # Simple evaluation - try all piece types
            forfuture = -INF
            for i in range(7):
                if myCount[i]:
                    typeCountForColor[currBotColor][i] += 1
                    test = work(map, i, deepth-1, forfuture, currBotColor, pop+1 if pop else 0)
                    if test > forfuture:
                        forfuture = test
                    typeCountForColor[currBotColor][i] -= 1

            if forfuture >= INF - 1:
                return INF - 1
            holenum += forfuture * W4
        else:
            # More complex evaluation - select top piece types
            TY = [0] * Type_num
            Work = [-INF] * Type_num
            forfuture = -INF

            # Evaluate all piece types
            for i in range(7):
                if myCount[i]:
                    typeCountForColor[currBotColor][i] += 1
                    test = work(map, i, 0, forfuture, currBotColor, pop+1 if pop else 0)

                    # Insert into top candidates
                    for j in range(Type_num):
                        if Work[j] < test:
                            for k in range(Type_num-1, j, -1):
                                TY[k] = TY[k-1]
                                Work[k] = Work[k-1]
                            Work[j] = test
                            TY[j] = i
                            break

                    typeCountForColor[currBotColor][i] -= 1

            # Evaluate top candidates in more depth
            for i in range(Type_num):
                if Work[i] > -INF + 1:
                    if Work[0] < 0:
                        if Work[i] < Work[0] * 1.05 - 1:
                            break
                    elif Work[i] < Work[0] * 0.95 - 1:
                        break

                    typeCountForColor[currBotColor][TY[i]] += 1
                    test = work(map, TY[i], deepth-1, forfuture, currBotColor, pop+1 if pop else 0)
                    if test > forfuture:
                        forfuture = test
                    typeCountForColor[currBotColor][TY[i]] -= 1

            if forfuture >= INF - 1:
                return INF - 1
            holenum += forfuture * W4

    # Subtract reward for cleared blocks from the new piece
    holenum -= pop * erodedShape

    return holenum

def work(_map, ty, deepth, forfuture, currBotColor, eRound):
    """Find the best move for the current piece"""
    global finalX, finalY, finalO

    bo = [[[False for _ in range(4)] for _ in range(MAPWIDTH + 2)] for _ in range(MAPHEIGHT + 2)]
    fx, fy, fo = -1, -1, -1
    MMH = INF

    # Initialize reachability
    initReachability(_map, bo, ty)

    if deepth > 0:
        # Evaluate multiple moves in depth
        work_scores = [INF] * Work_num
        XX = [0] * Work_num
        YY = [0] * Work_num
        OO = [0] * Work_num

        # Find candidate moves
        for y in range(1, MAPHEIGHT + 1):
            for x in range(1, MAPWIDTH + 1):
                for o in range(4):
                    if GO[ty][o] and onGround(_map, ty, x, y, o) and bo[y][x][o]:
                        test = Mavis(_map, ty, x, y, o, 0, MMH, currBotColor, eRound)

                        # Insert into top candidates
                        for i in range(Work_num):
                            if test < work_scores[i]:
                                for j in range(Work_num-1, i, -1):
                                    work_scores[j] = work_scores[j-1]
                                    XX[j] = XX[j-1]
                                    YY[j] = YY[j-1]
                                    OO[j] = OO[j-1]
                                work_scores[i] = test
                                XX[i] = x
                                YY[i] = y
                                OO[i] = o
                                break

        # Evaluate top candidates in depth
        for i in range(Work_num):
            if work_scores[i] < INF - 1:
                if work_scores[0] < 0:
                    if work_scores[i] > work_scores[0] * 0.8 + 10:
                        break
                elif work_scores[i] > work_scores[0] * 1.2 + 10:
                    break

                test = Mavis(_map, ty, XX[i], YY[i], OO[i], deepth, MMH, currBotColor, eRound)
                if test < MMH:
                    fx, fy, fo = XX[i], YY[i], OO[i]
                    MMH = test
                    if MMH < forfuture:
                        break
    else:
        # Simple evaluation - find best immediate move
        for y in range(1, MAPHEIGHT + 1):
            for x in range(1, MAPWIDTH + 1):
                for o in range(4):
                    if GO[ty][o] and onGround(_map, ty, x, y, o) and bo[y][x][o]:
                        test = Mavis(_map, ty, x, y, o, 0, INF, currBotColor, eRound)
                        if test < MMH:
                            fx, fy, fo = x, y, o
                            MMH = test
                            if MMH < forfuture:
                                break

    finalX, finalY, finalO = fx, fy, fo
    return MMH

def gain(_map, x, y, o, ty, _ty, currBotColor, DEEPTH):
    """Evaluate how good a move is for the opponent"""
    map = copy.deepcopy(_map)

    # Place the piece
    for i in range(4):
        tmpX = x + blockShape[ty][o][2*i]
        tmpY = y + blockShape[ty][o][2*i+1]
        map[tmpY][tmpX] = 3

    pop = 0

    # Check for completed rows
    for _y in range(1, MAPHEIGHT + 1):
        full = True
        for _x in range(1, MAPWIDTH + 1):
            if not map[_y][_x]:
                full = False
                break

        if full:
            pop += 1
            for _x in range(1, MAPWIDTH + 1):
                map[_y][_x] = -1

    # Clear completed rows
    for _x in range(1, MAPWIDTH + 1):
        p = 1
        for i in range(1, MAPHEIGHT + 1):
            if map[i][_x] != -1:
                map[p][_x] = map[i][_x]
                p += 1
        for i in range(p, MAPHEIGHT + 1):
            map[i][_x] = 0

    return work(map, _ty, DEEPTH, INF, currBotColor, pop+1 if pop else 0)

def blockForEnemy(_map, ty, DEEPTH):
    """Choose which piece to give to the opponent"""
    global finalX, finalY, finalO

    minCount = min(typeCountForColor[enemyColor])

    bo = [[[False for _ in range(4)] for _ in range(MAPWIDTH + 2)] for _ in range(MAPHEIGHT + 2)]
    initReachability(_map, bo, ty)

    # Find candidate moves for opponent
    work_scores = [INF] * Work_num
    XX = [0] * Work_num
    YY = [0] * Work_num
    OO = [0] * Work_num

    for y in range(1, MAPHEIGHT + 1):
        for x in range(1, MAPWIDTH + 1):
            for o in range(4):
                if onGround(_map, ty, x, y, o) and bo[y][x][o]:
                    test = Mavis(_map, ty, x, y, o, 0, INF, currBotColor, elimCombo[currBotColor])

                    # Insert into top candidates
                    for i in range(Work_num):
                        if test < work_scores[i]:
                            for j in range(Work_num-1, i, -1):
                                work_scores[j] = work_scores[j-1]
                                XX[j] = XX[j-1]
                                YY[j] = YY[j-1]
                                OO[j] = OO[j-1]
                            work_scores[i] = test
                            XX[i] = x
                            YY[i] = y
                            OO[i] = o
                            break

    # Evaluate which piece would be worst for opponent
    _mmh = -INF
    mmh = -1

    for _ty in range(7):
        if typeCountForColor[enemyColor][_ty] + 1 - minCount <= 2:
            typeCountForColor[enemyColor][_ty] += 1
            _MMH = INF

            # Evaluate opponent's best responses to this piece
            for i in range(Work_num):
                if work_scores[i] < INF - 1:
                    if work_scores[0] < 0:
                        if work_scores[i] > work_scores[0] * 0.8 + 10:
                            break
                    elif work_scores[i] > work_scores[0] * 1.2 + 10:
                        break

                    test = gain(_map, XX[i], YY[i], OO[i], ty, _ty, enemyColor, DEEPTH)
                    if test < _MMH:
                        _MMH = test

            # Choose piece that gives opponent worst position
            if _MMH > _mmh:
                _mmh = _MMH
                mmh = _ty

            typeCountForColor[enemyColor][_ty] -= 1

    return mmh

def main():
    """Main game loop"""
    global currBotColor, enemyColor, turnID, blockType, _finalX, _finalY, _finalO

    init()

    # Read initial game state
    turnID = int(sys.stdin.readline())
    parts = sys.stdin.readline().split()
    blockType = int(parts[0])
    currBotColor = int(parts[1])
    enemyColor = 1 - currBotColor
    nextTypeForColor[0] = blockType
    nextTypeForColor[1] = blockType
    typeCountForColor[0][blockType] += 1
    typeCountForColor[1][blockType] += 1

    # Reconstruct game history
    for i in range(1, turnID):
        # Read our previous move
        parts = sys.stdin.readline().split()
        currTypeForColor = [nextTypeForColor[0], nextTypeForColor[1]]
        blockType = int(parts[0])
        x = int(parts[1])
        y = int(parts[2])
        o = int(parts[3])

        # Place our piece
        myBlock = Tetris(currTypeForColor[currBotColor], currBotColor)
        myBlock.set(x, y, o).place()

        # Update piece counts
        typeCountForColor[enemyColor][blockType] += 1
        nextTypeForColor[enemyColor] = blockType

        # Read opponent's move
        parts = sys.stdin.readline().split()
        blockType = int(parts[0])
        x = int(parts[1])
        y = int(parts[2])
        o = int(parts[3])

        # Place opponent's piece
        enemyBlock = Tetris(currTypeForColor[enemyColor], enemyColor)
        enemyBlock.set(x, y, o).place()

        # Update piece counts
        typeCountForColor[currBotColor][blockType] += 1
        nextTypeForColor[currBotColor] = blockType

        # Process eliminations and transfers
        eliminate(0)
        eliminate(1)
        transfer()

    # Make decision for current turn
    work(gridInfo[currBotColor], blockType, M_DEEP, -INF, currBotColor, elimCombo[currBotColor])
    _finalX, _finalY, _finalO = finalX, finalY, finalO

    # Choose which piece to give to opponent
    enemy_block = blockForEnemy(gridInfo[enemyColor], nextTypeForColor[enemyColor], E_DEEP)

    # Output decision
    print(f"{enemy_block} {_finalX} {_finalY} {_finalO}")

if __name__ == "__main__":
    main()