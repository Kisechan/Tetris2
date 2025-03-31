import sys
import random
from copy import deepcopy

MAPWIDTH = 10
MAPHEIGHT = 20

currBotColor = 0
enemyColor = 1

gridInfo = [
    [[0]*(MAPWIDTH + 2) for _ in range(MAPHEIGHT + 2)],
    [[0]*(MAPWIDTH + 2) for _ in range(MAPHEIGHT + 2)]
]

trans = [
    [[0]*(MAPWIDTH + 2) for _ in range(6)],
    [[0]*(MAPWIDTH + 2) for _ in range(6)]
]
transCount = [0, 0]
maxHeight = [0, 0]
elimTotal = [0, 0]
elimCombo = [0, 0]
elimBonus = [0, 1, 3, 5, 7]
typeCountForColor = [
    [0]*7,
    [0]*7
]

blockShape = [
    [[0,0,1,0,-1,0,-1,-1], [0,0,0,1,0,-1,1,-1], [0,0,-1,0,1,0,1,1], [0,0,0,-1,0,1,-1,1]],
    [[0,0,-1,0,1,0,1,-1], [0,0,0,-1,0,1,1,1], [0,0,1,0,-1,0,-1,1], [0,0,0,1,0,-1,-1,-1]],
    [[0,0,1,0,0,-1,-1,-1], [0,0,0,1,1,0,1,-1], [0,0,-1,0,0,1,1,1], [0,0,0,-1,-1,0,-1,1]],
    [[0,0,-1,0,0,-1,1,-1], [0,0,0,-1,1,0,1,1], [0,0,1,0,0,1,-1,1], [0,0,0,1,-1,0,-1,-1]],
    [[0,0,-1,0,0,1,1,0], [0,0,0,-1,-1,0,0,1], [0,0,1,0,0,-1,-1,0], [0,0,0,1,1,0,0,-1]],
    [[0,0,0,-1,0,1,0,2], [0,0,1,0,-1,0,-2,0], [0,0,0,1,0,-1,0,-2], [0,0,-1,0,1,0,2,0]],
    [[0,0,0,1,-1,0,-1,1], [0,0,-1,0,0,-1,-1,-1], [0,0,0,-1,1,0,1,-1], [0,0,1,0,0,1,1,1]]
]

rotateBlank = [
    [[1,1,0,0], [-1,1,0,0], [-1,-1,0,0], [1,-1,0,0]],
    [[-1,-1,0,0], [1,-1,0,0], [1,1,0,0], [-1,1,0,0]],
    [[1,1,0,0], [-1,1,0,0], [-1,-1,0,0], [1,-1,0,0]],
    [[-1,-1,0,0], [1,-1,0,0], [1,1,0,0], [-1,1,0,0]],
    [[-1,-1,-1,1,1,1,0,0], [-1,-1,-1,1,1,-1,0,0], [-1,-1,1,1,1,-1,0,0], [-1,1,1,1,1,-1,0,0]],
    [[1,-1,-1,1,-2,1,-1,2,-2,2], [1,1,-1,-1,-2,-1,-1,-2,-2,-2], [-1,1,1,-1,2,-1,1,-2,2,-2], [-1,-1,1,1,2,1,1,2,2,2]],
    [[0,0], [0,0], [0,0], [0,0]]
]

class Tetris:
    def __init__(self, blockType, color):
        self.blockType = blockType
        self.color = color
        self.shape = blockShape[blockType]
        self.blockX = -1
        self.blockY = -1
        self.orientation = 0

    def set(self, x=-1, y=-1, o=-1):
        if x != -1: self.blockX = x
        if y != -1: self.blockY = y
        if o != -1: self.orientation = o
        return self

    def isValid(self, x=-1, y=-1, o=-1):
        x = x if x != -1 else self.blockX
        y = y if y != -1 else self.blockY
        o = o if o != -1 else self.orientation
        if o < 0 or o > 3:
            return False
        for i in range(4):
            tmpX = x + self.shape[o][2*i]
            tmpY = y + self.shape[o][2*i+1]
            if tmpX < 1 or tmpX > MAPWIDTH or tmpY < 1 or tmpY > MAPHEIGHT:
                return False
            if gridInfo[self.color][tmpY][tmpX] != 0:
                return False
        return True

    def onGround(self):
        if self.isValid() and not self.isValid(-1, self.blockY-1):
            return True
        return False

    def place(self):
        if not self.onGround():
            return False
        for i in range(4):
            x = self.blockX + self.shape[self.orientation][2*i]
            y = self.blockY + self.shape[self.orientation][2*i+1]
            gridInfo[self.color][y][x] = 2
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
            blank = rotateBlank[self.blockType][fromO]
            for i in range(0, len(blank), 2):
                if i >= len(blank): break
                bx = self.blockX + blank[i]
                by = self.blockY + blank[i+1]
                if bx == self.blockX and by == self.blockY:
                    break
                if gridInfo[self.color][by][bx] != 0:
                    return False
            fromO = (fromO + 1) % 4
        return True

def init():
    for i in range(MAPHEIGHT + 2):
        gridInfo[0][i][0] = gridInfo[0][i][MAPWIDTH+1] = -2
        gridInfo[1][i][0] = gridInfo[1][i][MAPWIDTH+1] = -2
    for i in range(MAPWIDTH + 2):
        gridInfo[0][0][i] = gridInfo[0][MAPHEIGHT+1][i] = -2
        gridInfo[1][0][i] = gridInfo[1][MAPHEIGHT+1][i] = -2

def checkDirectDropTo(color, blockType, x, y, o):
    shape = blockShape[blockType][o]
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
    count = 0
    hasBonus = 0
    maxHeight[color] = MAPHEIGHT
    newGrid = [[0]*(MAPWIDTH+2) for _ in range(MAPHEIGHT+2)]
    fullRows = []

    for y in range(1, MAPHEIGHT+1):
        full = all(gridInfo[color][y][x] != 0 for x in range(1, MAPWIDTH+1))
        empty = all(gridInfo[color][y][x] == 0 for x in range(1, MAPWIDTH+1))
        if full:
            fullRows.append(y)
        elif empty:
            maxHeight[color] = y-1
            break

    firstFull = True
    for y in fullRows:
        if firstFull and elimCombo[color] >= 2:
            trans[color][count] = [1 if gridInfo[color][y][x] == 1 else 0 for x in range(MAPWIDTH+2)]
            count += 1
            hasBonus = 1
        firstFull = False
        trans[color][count] = [1 if gridInfo[color][y][x] == 1 else 0 for x in range(MAPWIDTH+2)]
        count += 1

    transCount[color] = count
    elimCombo[color] = count // 2 if count > 0 else 0
    elimTotal[color] += elimBonus[count - hasBonus] if count - hasBonus < 5 else 7

    writeRow = MAPHEIGHT
    for y in range(MAPHEIGHT, 0, -1):
        if y not in fullRows:
            newGrid[writeRow] = gridInfo[color][y][:]
            writeRow -= 1

    for y in range(writeRow, 0, -1):
        newGrid[y] = [0]*(MAPWIDTH+2)

    gridInfo[color] = newGrid
    maxHeight[color] -= count - hasBonus

def transfer():
    color1, color2 = 0, 1
    if transCount[0] == 0 and transCount[1] == 0:
        return -1

    if transCount[0] == 0 or transCount[1] == 0:
        if transCount[0] == 0:
            color1, color2 = 1, 0
        h2 = maxHeight[color2] + transCount[color1]
        if h2 > MAPHEIGHT:
            return color2

        for y in range(h2, transCount[color1], -1):
            gridInfo[color2][y] = gridInfo[color2][y - transCount[color1]][:]
        for i in range(transCount[color1]):
            gridInfo[color2][i+1] = trans[color1][i][:]
        return -1

    else:
        h1 = maxHeight[color1] + transCount[color2]
        h2 = maxHeight[color2] + transCount[color1]
        if h1 > MAPHEIGHT: return color1
        if h2 > MAPHEIGHT: return color2

        temp = []
        for y in range(1, transCount[color1]+1):
            temp.append(trans[color1][y-1][:])
        for y in range(transCount[color1]+1, h2+1):
            temp.append(gridInfo[color2][y - transCount[color1]][:])
        gridInfo[color2] = [[-2]*(MAPWIDTH+2)] + temp + [[-2]*(MAPWIDTH+2)]*(MAPHEIGHT - h2)

        temp = []
        for y in range(1, transCount[color2]+1):
            temp.append(trans[color2][y-1][:])
        for y in range(transCount[color2]+1, h1+1):
            temp.append(gridInfo[color1][y - transCount[color2]][:])
        gridInfo[color1] = [[-2]*(MAPWIDTH+2)] + temp + [[-2]*(MAPWIDTH+2)]*(MAPHEIGHT - h1)

        return -1

def canPut(color, blockType):
    for y in range(MAPHEIGHT, 0, -1):
        for x in range(1, MAPWIDTH+1):
            for o in range(4):
                t = Tetris(blockType, color).set(x, y, o)
                if t.isValid() and checkDirectDropTo(color, blockType, x, y, o):
                    return True
    return False

def main():
    random.seed()
    init()

    lines = sys.stdin.read().split('\n')
    ptr = 0

    turnID = int(lines[ptr])
    ptr +=1

    first = list(map(int, lines[ptr].split()))
    ptr +=1
    blockType, currBotColor = first[0], first[1]
    enemyColor = 1 - currBotColor
    nextTypeForColor = [blockType, blockType]
    typeCountForColor[0][blockType] +=1
    typeCountForColor[1][blockType] +=1

    for _ in range(1, turnID):
        currType = [nextTypeForColor[0], nextTypeForColor[1]]

        myAct = list(map(int, lines[ptr].split()))
        ptr +=1
        bt, x, y, o = myAct
        myBlock = Tetris(currType[currBotColor], currBotColor).set(x, y, o)
        myBlock.place()
        typeCountForColor[enemyColor][bt] +=1
        nextTypeForColor[enemyColor] = bt

        enemyAct = list(map(int, lines[ptr].split()))
        ptr +=1
        bt, x, y, o = enemyAct
        enemyBlock = Tetris(currType[enemyColor], enemyColor).set(x, y, o)
        enemyBlock.place()
        typeCountForColor[currBotColor][bt] +=1
        nextTypeForColor[currBotColor] = bt

        eliminate(0)
        eliminate(1)
        transfer()

    block = Tetris(nextTypeForColor[currBotColor], currBotColor)
    finalX, finalY, finalO = 1, 1, 0
    found = False
    for y in range(1, MAPHEIGHT+1):
        for x in range(1, MAPWIDTH+1):
            for o in range(4):
                if block.set(x, y, o).isValid() and checkDirectDropTo(currBotColor, block.blockType, x, y, o):
                    finalX, finalY, finalO = x, y, o
                    found = True
                    break
            if found: break
        if found: break

    maxCount = max(typeCountForColor[enemyColor])
    minCount = min(typeCountForColor[enemyColor])
    if maxCount - minCount == 2:
        for bt in range(7):
            if typeCountForColor[enemyColor][bt] != maxCount:
                blockForEnemy = bt
                break
    else:
        blockForEnemy = random.randint(0,6)

    print(f"{blockForEnemy} {finalX} {finalY} {finalO}")

if __name__ == "__main__":
    main()