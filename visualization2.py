import matplotlib.pyplot as plt

#折线图
x = [1,2,3,4,5]#点的横坐标
x_labels = ["zero-shot", "expert", "zero-shot-CoT", "few-shot", "expert-few-shot-CoT"]
# Capability
k1 = [0.3875,0.425,0.4875,0.325,0.846]#线1的纵坐标
k2 = [0.2848,0.2948,0.2884,0.4882,0.6834]#线2的纵坐标
k3 = [0.2042,0.236,0.3237,0.3057,0.8125]#线1的纵坐标
k4 = [0.1156,0.1834,0.1653,0.4024,0.5607]#线2的纵坐标
k5 = [0.1013,0.1243,0.1569,0.2965,0.4091]#线2的纵坐标
'''# Accuracy
k1 = [0.27,0.2985,0.2685,0.219,0.4795]#线1的纵坐标
k2 = [0.1932,0.2032,0.187,0.2074,0.3954]#线2的纵坐标
k3 = [0.169,0.1972,0.2613,0.2432,0.6473]#线1的纵坐标
k4 = [0.0753,0.1093,0.0876,0.2027,0.2926]#线2的纵坐标
k5 = [0.0552,0.069,0.0851,0.1407,0.1923]#线2的纵坐标
# Hallucination Drop
k1 = [0.3091,0.2986,0.4484,0.2882,0.4343]#线1的纵坐标
k2 = [0.5977,0.4570,0.5592,0.6239,0.4384]#线2的纵坐标
k3 = [0.1834,0.2487,0.3012,0.2068,0.2077]#线1的纵坐标
k4 = [0.3385,0.4963,0.5657,0.4782,0.5177]#线2的纵坐标
k5 = [0.5789,0.6557,0.5989,0.5365,0.5236]#线2的纵坐标
'''
plt.plot(x,k1,'s-',color = 'r',label="Arithmetic Tasks")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="Spatial Relationship Tasks")#o-:圆形
plt.plot(x,k3,'s-',color = 'b',label="Domain Knowledge Literal Reasoning Tasks")#s-:方形
plt.plot(x,k4,'o-',color = 'y',label="Common Property Retrieval Tasks")#o-:圆形
plt.plot(x,k5,'s-',color = 'k',label="Uncommon Property Retrieval Tasks")#s-:方形
plt.xlabel("Prompt Engineering Methods")#横坐标名字
plt.ylabel("Capability")#纵坐标名字
plt.xticks(x, x_labels)
plt.ylim([0, 1])
plt.legend(loc = "best")#图例
plt.show()