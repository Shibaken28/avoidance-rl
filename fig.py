import matplotlib.pyplot as plt

FILE_NAME = [
    "./vec/20240118_142037/",
    "./cnn/20240118_143716/",
    "./cnn2/20240118_152511/",
    "./duel/20240118_152606/",
    "./duel2/20240119_090341/",
]

alpha = ["A","B","C","D","E"]


for x, file_name in enumerate(FILE_NAME):
    # 0,1の数をカウント
    print(file_name)
    cnt = [0, 0]
    with open(file_name+"result_test.csv", "r") as f:
        for line in f:
            a = int(line)
            cnt[a] += 1
    print(cnt)
    print(cnt[1] / (cnt[0] + cnt[1]))
    
    res = []
    with open(file_name+"result_all.csv", "r") as f:
        for line in f:
            res.append(int(line))
        # 直近500件の平均を取る
        avg = []
        sum = 0
        w = 1000
        for i in range(len(res)):
            sum += res[i]
            avg.append(sum/w)
            if i >= w:
                sum -= res[i-w]
        # グラフを描画、凡例はalpha[x]
        # モノクロになる可能性があるため、線の種類を変えておく
        if x in [0, 1,3]:
            plt.plot(avg, label=alpha[x])#, linestyle=["-", "--", "--", ":", ":"][x])
            # plt.legend()
            
                    
            plt.xlabel("episode")
            plt.ylabel("win rate")
            plt.savefig("./result.png")