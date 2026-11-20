import numpy as np

def main():
	np.random.seed(42)
	#定义三个簇的中心点
	centers=np.array([[0,0],[10,10],[20,0]])
	n_per_cluster=2000 #每个簇中二维点数量
	points=[] #初始化数据点列表，存放所有数据点
	#遍历生成每个簇的所有二维点
	for c in centers:
		#在簇中心基础上随机生成满足正态分布的2000个二维数据点
		pts=c+np.random.randn(n_per_cluster,2)*1.5	
		points.append(pts) #将生成的二维数据点加入数据点列表中
	data = np.vstack(points) #按列存放每个数据点坐标(x,y)
	
	#permutation打乱数据点顺序
	idx=np.random.permutation(len(data))  #打乱索引
	data=data[idx] #打乱数据
	np.savetxt("points.csv",data,delimiter=",",fmt="%.6f") #将数据保存至csv文件中，并设置分隔符为','
	
	print("生成数据：",data.shape,"已保存到data/points.csv")
	
#主程序入口
if __name__=="__main__":
	main()
	
