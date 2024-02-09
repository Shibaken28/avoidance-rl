"""
to do
- rewaradの平均も出力するようにする
"""

import copy
from collections import deque
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from PIL import Image

# グラフ描画
import matplotlib.pyplot as plt
# 日付取得
import datetime
# csv出力
import csv
# ディレクトリ作成
import os
# 画像保存
import time



import string
# idは年月日_時分秒とする
now = datetime.datetime.now()
id = now.strftime('%Y%m%d_%H%M%S')
print(id)
# ディレクトリ作成
os.mkdir("./"+id)
os.mkdir("./"+id+"/img")
os.mkdir("./"+id+"/csv")

ACTION_SPACE = 3 # 行動の種類
STATE_SPACE = (1+10)*2 # エージェントと障害物の座標の数

# リサイズ後の画像サイズ

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 30


class Abstacle:
    def __init__(self, x, y, size, speedX, speedY):
        self.x = x
        self.y = y
        self.size = size
        self.speedX = speedX
        self.speedY = speedY
        self.pos_history = []
        self.pos_history.append((self.x, self.y))

        # 蛇行
        self.snake = False
        self.snake_dir = 1 # 蛇行する方向
        self.snake_time = 0 # 蛇行する時間

    def step(self, dt):
        if self.snake:
            self.snake_time += 1
            self.x += self.snake_dir * dt * 0.3
            if self.snake_time >= 15:
                self.snake = False
                self.snake_time = 0
        self.x += self.speedX * dt
        self.y += self.speedY * dt
        self.pos_history.append((self.x, self.y))
        # 確率で蛇行
        if random.randint(0, 50) == 0:
            self.startSnake()


    def startSnake(self):
        self.snake = True
        self.snake_dir = random.choice([-1, 1])
        self.snake_time = 0

    def isCollision(self, x, y, size):
        distance = np.sqrt((self.x-x)**2 + (self.y-y)**2)
        if distance < self.size + size:
            print(f"障害物の座標:({self.x}, {self.y}), 自機の座標:({x}, {y}), 距離が{distance}なので衝突")
            return True
        else:
            return False

class Game:
    """
    500カウント以内に目的地につけば成功
    敵は3体
    state:
        自分の座標
        敵の座標
    action:
        右に行く(x+1), 左に行く(x-1), そのまま(x)
    reward:
        敵に当たったら-1, それ以外は0
        ゲーム終了時に成功なら+1
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        # クリアフラグ
        self.win = False
        self.total_reward = 0

        # 敵の軌道
        self.game_limit = 120
        self.width = 30
        self.height = 60
        self.time = 0

        # 自機の初期位置
        self.player_pos = (self.width/2, self.height-1)
        self.player_size = 1
        self.player_pos_history = []
        self.player_pos_history.append(self.player_pos)


        self.enemy_list = []
        self.enemy_num = 10
        for i in range(self.enemy_num):
            # 敵の初期位置
            x = random.randint(0, self.width)
            y = self.height - 20 - i*10
            size = 2 #random.randint(1, 10)
            speedX = 0 # random.randint(-10, 10)/100
            speedY = 1
            self.enemy_list.append(Abstacle(x, y, size, speedX, speedY))

        return self.states(), 0

    def states(self):
        
        # 現在の状態を画像として返す
        # 画像サイズはwidth*height
        # 画像の背景は黒、自機は白、敵は赤
        img = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        # 1点を描画
        if self.player_pos[0] >= 0 and self.player_pos[0] < self.width and self.player_pos[1] >= 0 and self.player_pos[1] < self.height:
            img.putpixel((int(self.player_pos[0]), int(self.player_pos[1])), (0, 255, 0))
        for enemy in self.enemy_list:
            if enemy.x >= 0 and enemy.x < self.width and enemy.y >= 0 and enemy.y < self.height:
                img.putpixel((int(enemy.x), int(enemy.y)), (255, 0, 0))


        # デバッグ用に画像を保存
        # img.save("img.png")

        # 0.1秒まつ
        # time.sleep(0.1)


        # 画像のリサイズと正規化
        transform = T.Compose([T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), T.ToTensor()])
        img = transform(img).unsqueeze(0)

        return img


    def info(self):
        info = []
        info.append(self.height)
        info.append(self.width)
        # 自分の位置
        info.append(self.player_size)
        info.append(len(self.player_pos_history))
        for pos in self.player_pos_history:
            info.append(pos[0])
            info.append(pos[1])
        # 敵の位置
        info.append(len(self.enemy_list))
        for enemy in self.enemy_list:
            info.append(enemy.size)
            info.append(len(enemy.pos_history))
            for pos in enemy.pos_history:
                info.append(pos[0])
                info.append(pos[1])
        if self.win:
            info.append(1)
        else:
            info.append(0)
        info.append(self.total_reward)
        return info
    


    def step(self, action):
        self.time += 1
        # actionを実行
        reward = 0
        done = False
        # (0,1,2,3) = (そのまま, 右前, 左前, 前)
        speed = 1
        # 角度
        angle = -90
        if action == 0:
            # 右に行く
            angle += 30
        elif action == 1:
            # 左に行く
            angle -= 30
        elif action == 2:
            # 前に行く
            pass
        else:
            print("error")
            
        # 角度をラジアンに変換
        rad = np.radians(angle)
        # 速度を計算
        speedX = speed * np.cos(rad)
        speedY = speed * np.sin(rad)
        # 位置を更新
        self.player_pos = (self.player_pos[0] + speedX, self.player_pos[1] + speedY)

        # 画面買いに行ったら戻す
        if self.player_pos[0] < 0:
            self.player_pos = (0, self.player_pos[1])
        elif self.player_pos[0] >= self.width:
            self.player_pos = (self.width-1, self.player_pos[1])
        if self.player_pos[1] < 0:
            self.player_pos = (self.player_pos[0], 0)
        elif self.player_pos[1] >= self.height:
            self.player_pos = (self.player_pos[0], self.height-1)


        self.player_pos_history.append(self.player_pos)

        # 敵の位置を更新
        for enemy in self.enemy_list:
            enemy.step(1)
        # 報酬を計算
        if self.isCollision():
            print(f"時刻{self.time}")
            print("障害物に衝突してしまいました")
            reward = -1
            done = True
        # 上のほうへ行ったら成功
        elif self.player_pos[1] < 20:
            print(f"時刻{self.time}")
            print("ゴール領域に到達")
            reward = 1
            done = True
            self.win = True
        elif self.time == self.game_limit:
            reward = -1
            done = True
        else:
            pass
            
        self.total_reward += reward
        return self.states(), reward, done, {}, self.win



    def isCollision(self):
        # 敵と自機の距離を計算
        for enemy in self.enemy_list:
            if enemy.isCollision(self.player_pos[0], self.player_pos[1], self.player_size):
                return True
        return False


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int64))
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))

        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, num_actions):
        super(QNet, self).__init__()
        # 畳み込み
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # 全結合
        self.fc_val1 = nn.Linear(self.feature_size(), 256)
        self.fc_val2 = nn.Linear(256, 1)
        # Dueling Network
        self.fc_adv1 = nn.Linear(self.feature_size(), 256)
        self.fc_adv2 = nn.Linear(256, num_actions)

    def feature_size(self):
        return self.conv2(self.conv1(torch.zeros(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH))).view(1, -1).size(1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        val = F.relu(self.fc_val1(x))
        val = self.fc_val2(val)

        adv = F.relu(self.fc_adv1(x))
        adv = self.fc_adv2(adv)

        q_values = val + (adv - adv.mean(dim=1, keepdim=True))

        return q_values
    
    """

    def __init__(self, num_actions):
        # maxpoolingを使う
        super(QNet, self).__init__()
        # 畳み込み
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        # 全結合
        print(self.feature_size())
        self.fc1 = nn.Linear(self.feature_size(), 256)
        self.fc2 = nn.Linear(256, num_actions)

    def feature_size(self):
        return self.conv2(self.pool1(self.conv1(torch.zeros(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)))).view(1, -1).size(1)
    
    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    """


class DQNAgent:
    def __init__(self):
        self.gamma = 0.95
        self.lr = 0.0005
        self.epsilon = 0.05
        self.buffer_size = 50000
        self.batch_size = 32
        self.action_size = ACTION_SPACE

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state, eps = 0.01):
        self.epsilon = eps
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            # state_tensor = torch.tensor(state).unsqueeze(0)  # バッチ次元を追加
            # qs = self.qnet(state_tensor)
            # state = state.squeeze(0)  # バッチ次元を削除
            # state = state.permute(1, 2, 0)  # チャンネルの次元を最後に移動
            qs = self.qnet(state)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        # [32, 1, 3, 84, 84] を [32, 3, 84, 84] に変換
        #print("before ", state.shape)
        state = state.squeeze(1)
        # print("after ", state.shape)
        next_state = next_state.squeeze(1)
    
        qs = self.qnet(state)

        q = qs[np.arange(len(action)), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]

        next_q.detach()
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


# 途中経過を保存
def draw_graph(result):
    # 直近300回の成功率
    rate = []
    for i in range(len(result)):
        if i < 300:
            rate.append(sum(result[:i+1])/(i+1))
        else:
            rate.append(sum(result[i-300:i+1])/300)
    plt.plot(rate)
    plt.xlabel('episode')
    plt.ylabel('rate')
    # y軸の範囲を0~1にする
    plt.ylim(0, 1)
    # 現在の日付+時間+エピソード数をファイル名にする
    now = datetime.datetime.now()
    path = './'+id+'/img/'
    # YYYYMMDD_HHMMSS_episode.png
    filename = "result_"+id+"_"+ str(episode+1).zfill(6) + '.png'
    print(filename)
    plt.savefig(path + filename)
    plt.close()

def output_csv(episode):
    # 現在の日付+時間+エピソード数をファイル名にする
    now = datetime.datetime.now()
    path = './'+id+'/csv/'
    # YYYYMMDD_HHMMSS_episode.csv
    # エピソード数は6桁で0埋め
    filename = "result_"+id+"_" + str(episode+1).zfill(6) + '.csv'
    print(filename)
    # csvファイルを開く
    info = env.info()
    with open(path + filename, 'w') as f:
        # ヘッダーを指定する
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(info)
    # ファイルを閉じる
    f.close()
    print("save csv")    



episodes = 10000
sync_interval = 20
env = Game()
agent = DQNAgent()
reward_history = []
result = []

# ビジュアライズするために状態を保存

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    win = False
    while not done:
    
        action = agent.get_action(state)

        next_state, reward, done, info, win  = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()

    result.append(win)
    reward_history.append(total_reward)
    print("episode :{}, win : {}, total reward : {}".format(episode, win, total_reward))
    output_csv(episode)
    # 平方数の時にグラフを描画
    r = int(np.sqrt(episode))
    if r*r == episode:
        draw_graph(result)
        
draw_graph(result)
    
# resultをcsvに出力
with open('./'+id+'/result_all.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for i in range(len(result)):
        if result[i]:
            writer.writerow([1])
        else:
            writer.writerow([0])
f.close()

test_num = 1000
test_result = []
# 学習を行わないで、テストを行う
for episode in range(test_num):
    state, _ = env.reset()
    done = False
    total_reward = 0
    win = False
    while not done:
        action = agent.get_action(state, eps=0.01)
        next_state, reward, done, info, win  = env.step(action)
        state = next_state
        total_reward += reward
    test_result.append(win)
    print("test-episode :{}, win : {}, total reward : {}".format(episode, win, total_reward))
    output_csv(episode+episodes)

# test_resultをcsvに出力
with open('./'+id+'/result_test.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for i in range(len(test_result)):
        if test_result[i]:
            writer.writerow([1])
        else:
            writer.writerow([0])
f.close()
