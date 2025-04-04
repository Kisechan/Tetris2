import numpy as np
import random
from copy import deepcopy

class Tetris2Env:

    MAPWIDTH = 10
    MAPHEIGHT = 20

    # 方块形状定义
    BLOCK_SHAPES = [
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

    # 旋转检查空白格
    ROTATE_BLANKS = [
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

    ELIM_BONUS = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    class Tetris:
        """方块类"""
        def __init__(self, env, block_type, color):
            self.env = env
            self.block_type = block_type
            self.color = color
            self.shape = env.BLOCK_SHAPES[block_type]
            self.x = -1
            self.y = -1
            self.orientation = 0

        def set_pos(self, x=-1, y=-1, o=-1):
            if x != -1: self.x = x
            if y != -1: self.y = y
            if o != -1: self.orientation = o
            return self

        def is_valid(self, x=-1, y=-1, o=-1):
            x = x if x != -1 else self.x
            y = y if y != -1 else self.y
            o = o if o != -1 else self.orientation

            if o < 0 or o > 3:
                return False

            for i in range(4):
                dx = x + self.shape[o][2*i]
                dy = y + self.shape[o][2*i+1]
                if dx < 1 or dx > self.env.MAPWIDTH or dy < 1 or dy > self.env.MAPHEIGHT:
                    return False
                if self.env.grid_info[self.color][dy][dx] != 0:
                    return False
            return True

        def on_ground(self):
            return self.is_valid() and not self.is_valid(-1, self.y-1)

        def place(self):
            if not self.on_ground():
                return False

            # 标记旧方块
            for y in range(1, self.env.MAPHEIGHT):
                for x in range(1, self.env.MAPWIDTH+1):
                    if self.env.grid_info[self.color][y][x] == 2:
                        self.env.grid_info[self.color][y][x] = 1

            # 放置新方块
            for i in range(4):
                x = self.x + self.shape[self.orientation][2*i]
                y = self.y + self.shape[self.orientation][2*i+1]
                self.env.grid_info[self.color][y][x] = 2
            return True

        def can_rotate(self, new_o):
            if new_o < 0 or new_o > 3:
                return False
            if self.orientation == new_o:
                return True

            from_o = self.orientation
            while True:
                if not self.is_valid(-1, -1, from_o):
                    return False
                if from_o == new_o:
                    break

                blank = self.env.ROTATE_BLANKS[self.block_type][from_o]
                for i in range(0, len(blank), 2):
                    if i >= len(blank): break
                    bx = self.x + blank[i]
                    by = self.y + blank[i+1]
                    if bx == self.x and by == self.y:
                        break
                    if self.env.grid_info[self.color][by][bx] != 0:
                        return False

                from_o = (from_o + 1) % 4
            return True

    def __init__(self):
        self.reset()

    def reset(self):
        self.grid_info = [
            [[0]*(self.MAPWIDTH + 2) for _ in range(self.MAPHEIGHT + 2)],
            [[0]*(self.MAPWIDTH + 2) for _ in range(self.MAPHEIGHT + 2)]
        ]

        self.trans = [
            [[0]*(self.MAPWIDTH + 2) for _ in range(6)],
            [[0]*(self.MAPWIDTH + 2) for _ in range(6)]
        ]
        self.trans_count = [0, 0]

        self.max_height = [0, 0]
        self.elim_total = [0, 0]
        self.elim_combo = [0, 0]

        self.type_count = [
            [0]*7,
            [0]*7
        ]

        self.current_type = [0, 0]
        self.next_type = [0, 0]

        # 初始化边界
        self._init_boundaries()

        # 随机初始化第一个方块
        block_type = random.randint(0, 6)
        self.current_type = [block_type, block_type]
        self.next_type = [block_type, block_type]
        self.type_count[0][block_type] += 1
        self.type_count[1][block_type] += 1

        self.turn = 0
        self.done = False

    def _init_boundaries(self):
        """初始化地图边界"""
        for i in range(self.MAPHEIGHT + 2):
            self.grid_info[0][i][0] = self.grid_info[0][i][self.MAPWIDTH+1] = -2
            self.grid_info[1][i][0] = self.grid_info[1][i][self.MAPWIDTH+1] = -2
        for i in range(self.MAPWIDTH + 2):
            self.grid_info[0][0][i] = self.grid_info[0][self.MAPHEIGHT+1][i] = -2
            self.grid_info[1][0][i] = self.grid_info[1][self.MAPHEIGHT+1][i] = -2

    def check_direct_drop(self, color, block_type, x, y, o):
        """检查是否能从顶部直接落到指定位置"""
        shape = self.BLOCK_SHAPES[block_type][o]
        for cy in range(y, self.MAPHEIGHT + 1):
            for i in range(4):
                dx = x + shape[2*i]
                dy = cy + shape[2*i+1]
                if dy > self.MAPHEIGHT:
                    continue
                if dy < 1 or dx < 1 or dx > self.MAPWIDTH or self.grid_info[color][dy][dx] != 0:
                    return False
        return True

    def eliminate(self, color):
        """消除满行并处理转移行"""
        count = 0
        has_bonus = 0
        self.max_height[color] = self.MAPHEIGHT
        full_rows = []

        # 找出满行
        for y in range(1, self.MAPHEIGHT+1):
            full = all(self.grid_info[color][y][x] != 0 for x in range(1, self.MAPWIDTH+1))
            empty = all(self.grid_info[color][y][x] == 0 for x in range(1, self.MAPWIDTH+1))
            if full:
                full_rows.append(y)
            elif empty:
                self.max_height[color] = y-1
                break

        # 处理满行
        first_full = True
        for y in full_rows:
            if first_full and self.elim_combo[color] >= 2:
                self.trans[color][count] = [
                    self.grid_info[color][y][x] if self.grid_info[color][y][x] in (1, -2) else 0
                    for x in range(self.MAPWIDTH+2)
                ]
                count += 1
                has_bonus = 1
            first_full = False

            self.trans[color][count] = [
                self.grid_info[color][y][x] if self.grid_info[color][y][x] in (1, -2) else 0
                for x in range(self.MAPWIDTH+2)
            ]
            count += 1

        self.trans_count[color] = count
        self.elim_combo[color] = count // 2 if count > 0 else 0
        self.elim_total[color] += self.ELIM_BONUS[count - has_bonus] if count - has_bonus < 5 else 7

        if count == 0:
            return

        # 重建地图
        new_grid = []
        write_row = self.MAPHEIGHT
        for y in range(1, self.MAPHEIGHT):
            if y not in full_rows:
                new_grid.append(self.grid_info[color][y][:])
                write_row -= 1

        # 填充空行
        while len(new_grid) <= self.MAPHEIGHT:
            new_grid.append([-2] + [0]*self.MAPWIDTH + [-2])

        self.grid_info[color] = new_grid
        self.max_height[color] = self.MAPHEIGHT - write_row

    def transfer_lines(self):
        """处理行转移"""
        if self.trans_count[0] == 0 and self.trans_count[1] == 0:
            return -1

        # 只有一方有转移行
        if self.trans_count[0] == 0 or self.trans_count[1] == 0:
            color1, color2 = (1, 0) if self.trans_count[0] == 0 else (0, 1)

            h2 = self.max_height[color2] + self.trans_count[color1]
            if h2 > self.MAPHEIGHT:
                return color2

            # 移动行
            for y in range(h2, self.trans_count[color1], -1):
                self.grid_info[color2][y] = self.grid_info[color2][y - self.trans_count[color1]][:]

            # 添加转移行
            for i in range(self.trans_count[color1]):
                self.grid_info[color2][i+1] = self.trans[color1][i][:]

            return -1

        # 双方都有转移行
        else:
            h1 = self.max_height[0] + self.trans_count[1]
            h2 = self.max_height[1] + self.trans_count[0]

            if h1 > self.MAPHEIGHT: return 0
            if h2 > self.MAPHEIGHT: return 1

            # 处理玩家1的场地
            temp = []
            for y in range(1, self.trans_count[0]+1):
                temp.append(self.trans[0][y-1][:])
            for y in range(self.trans_count[0]+1, h2+1):
                temp.append(self.grid_info[1][y - self.trans_count[0]][:])
            self.grid_info[1] = [[-2]*(self.MAPWIDTH+2)] + temp + [[-2]*(self.MAPWIDTH+2)]*(self.MAPHEIGHT - h2)

            # 处理玩家0的场地
            temp = []
            for y in range(1, self.trans_count[1]+1):
                temp.append(self.trans[1][y-1][:])
            for y in range(self.trans_count[1]+1, h1+1):
                temp.append(self.grid_info[0][y - self.trans_count[1]][:])
            self.grid_info[0] = [[-2]*(self.MAPWIDTH+2)] + temp + [[-2]*(self.MAPWIDTH+2)]*(self.MAPHEIGHT - h1)

            return -1

    def can_place_block(self, color, block_type):
        """检查是否能放置指定类型的方块"""
        for y in range(self.MAPHEIGHT, 0, -1):
            for x in range(1, self.MAPWIDTH+1):
                for o in range(4):
                    if (self.Tetris(self, block_type, color)
                            .set_pos(x, y, o)
                            .is_valid() and
                            self.check_direct_drop(color, block_type, x, y, o)):
                        return True
        return False

    def step(self, color, action):
        """
        执行一个动作
        action: (block_for_enemy, x, y, o)
        """
        block_for_enemy, x, y, o = action

        # 检查动作合法性
        if not (1 <= x <= self.MAPWIDTH and 1 <= y <= self.MAPHEIGHT and 0 <= o <= 3):
            self.done = True
            print(f"x={x}, y={y}, o={o}, 动作不合法")
            return -10, True

        # 检查方块类型选择是否合法
        enemy_counts = self.type_count[1 - color]
        min_count = min(enemy_counts)
        if enemy_counts[block_for_enemy] > min_count + 2:
            self.done = True
            print(f"方块类型不合法")
            return -10, True

        # 放置当前方块
        block = self.Tetris(self, self.current_type[color], color)
        if not block.set_pos(x, y, o).is_valid() or not block.place():
            self.done = True
            return -10, True

        # 更新方块类型
        self.type_count[1 - color][block_for_enemy] += 1
        self.current_type[color] = self.next_type[color]
        self.next_type[color] = block_for_enemy

        # 消除行
        self.eliminate(color)

        # 行转移
        loser = self.transfer_lines()
        if loser != -1:
            self.done = True
            return 10 if loser != color else -10, True

        # 检查游戏是否结束
        if not self.can_place_block(color, self.current_type[color]):
            self.done = True
            return -10, True

        # 计算奖励
        reward = self.elim_combo[color] * 0.5  # 连消奖励
        reward -= self.max_height[color] * 0.01  # 高度惩罚

        self.turn += 1
        return reward, self.done

    def get_state(self, color):
        """获取当前状态表示"""
        grid = []
        for y in range(1, 21):  # 第1到20行
            for x in range(1, 11):  # 第1到10列
                grid.append(self.grid_info[color][y][x])
        grid = np.array(grid)

        # 当前方块类型 (one-hot)
        current_block = np.zeros(7)
        current_block[self.current_type[color]] = 1

        # 对方方块统计
        enemy_stats = np.array(self.type_count[1 - color])

        # 连消计数和最大高度
        combo = np.array([self.elim_combo[color]])
        height = np.array([self.max_height[color]])

        return np.concatenate([grid, current_block, enemy_stats, combo, height])