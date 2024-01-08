import matplotlib.pyplot as plt

#折线图
x = [1,2,3,4,5]#点的横坐标
x_labels = ["zero-shot", "expert", "zero-shot-CoT", "few-shot", "expert-few-shot-CoT"]
'''# Capability
k1 = [0.567,0.6113,0.5777,0.6,0.9447]#线1的纵坐标
k2 = [0.0636,0.0848,0.0989,0.3588,0.4221]#线2的纵坐标
k3 = [0.2566,0.2932,0.3618,0.265,0.8116]#线1的纵坐标
k4 = [0.1976,0.256,0.3107,0.2966,0.6787]#线2的纵坐标
# Accuracy
k1 = [0.387,0.4183,0.3763,0.347,0.504]#线1的纵坐标
k2 = [0.0232,0.0316,0.035,0.1368,0.1778]#线2的纵坐标
k3 = [0.2128,0.24,0.2756,0.2222,0.6742]#线1的纵坐标
k4 = [0.1557,0.1996,0.2246,0.2233,0.4673]#线2的纵坐标
'''# Hallucination Drop
k1 = [0.3361,0.3176,0.3619,0.4149,0.4766]#线1的纵坐标
k2 = [0.6042,0.6735,0.6619,0.6309,0.5808]#线2的纵坐标
k3 = [0.1814,0.1799,0.2481,0.1567,0.1661]#线1的纵坐标
k4 = [0.2250,0.2933,0.4383,0.2812,0.2970]#线2的纵坐标

plt.plot(x,k1,'s-',color = 'r',label="Numerical Answer by Logic")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="Numerical Answer by Experimental Data")#o-:圆形
plt.plot(x,k3,'s-',color = 'b',label="Verbal Answer by Logic")#s-:方形
plt.plot(x,k4,'o-',color = 'y',label="Verbal Answer by Experimental Data")#o-:圆形
plt.xlabel("Prompt Engineering Methods")#横坐标名字
plt.ylabel("Hallucination Drop")#纵坐标名字
plt.xticks(x, x_labels)
plt.ylim([0, 1])
plt.legend(loc = "best")#图例
plt.show()