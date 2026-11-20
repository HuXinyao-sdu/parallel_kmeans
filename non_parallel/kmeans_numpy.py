import numpy as np

#从指定路径读取数据
def load_data(path):
	#读取以逗号分割的文本文件
	return np.loadtxt(path,delimiter=",") 

#初始化k个聚类中心
def init_centroids(data,k,seed=42):
	#创建带固定种子的随机数生成器
	rng=np.random.RandomState(seed)
	#无放回抽取k个数据点索引作为初始聚类中心索引
	idx=rng.choice(len(data),size=k,replace=False) 
	#抽取索引对应数据点作为聚类中心并复制一份
	return data[idx].copy()
	
def assign_labels(data,centroids):
	#计算每个点到聚类中心的欧式距离，取最小值对应聚类ID
	#data:(N,D),centroids:(k,D)
	#通过广播机制进行计算，并沿D维求和=>(N,K)距离平方矩阵
	dists=np.sum((data[:,None,:]-centroids[None,:,:])**2,axis=2)
	#对每行取距离最小的中心索引，返回聚类标签
	return np.argmin(dists,axis=1)

#根据当前标签重新计算各簇中心
def update_centroids(data,labels,k):
	new_centroids = np.zeros((k,data.shape[1])) #(K,D)全零
	#遍历每个簇
	for i in range(k):
		pts=data[labels==i] #加载当前簇中数据点列表
		if len(pts)>0: #不是空簇，存在数据点
			new_centroids[i]=pts.mean(axis=0) #计算当前簇平均值
		else:#为空簇
			new_centroids[i]=np.nan #用NAN标记当前簇中心
	#返回新簇中心
	return new_centroids

#主函数
def kmeans(data,k,max_iter=100,tol=1e-6,seed=42):
	centroids=init_centroids(data,k,seed)
	labels=None
	for it in range(max_iter):
		labels=assign_labels(data,centroids)
		new_centroids=update_centroids(data,labels,k)
		#空簇处理：保留旧中心
		empty=np.isnan(new_centroids[:,0])
		new_centroids[empty]=centroids[empty]
		shift=np.sum((new_centroids-centroids)**2) #所有中心移动量平方和
		centroids=new_centroids
		print(f"[iter {it}] shift={shift:.8f}")
		if shift<tol: #基本不变，提前终止循环
			break
	
	#最终分配一次标签，确保labels与最终中心一致
	labels=assign_labels(data,centroids)
	return centroids,labels

#结果保存函数
def save_result(centroids,labels,out_prefix):
	np.savetxt(out_prefix+"_centroids.csv",centroids,delimiter=",",fmt="%.6f")
	np.savetxt(out_prefix+"_labels.csv",labels,fmt="%d")
	print("结果已保存至",out_prefix+"_centroids.csv",out_prefix+"_labels.csv")

if __name__=="__main__":
	data=load_data("../data/points.csv")
	k=3
	
	#迭代得到最终结果
	centroids,labels=kmeans(data,k)
	#按簇中心进行排序
	order=np.lexsort((centroids[:,1],centroids[:,0]))
	centroids=centroids[order]
	#标签重映射
	mapping={old:new for new,old in enumerate(order)}
	labels=np.array([mapping[i] for i in labels])
	#保存最终结果
	save_result(centroids,labels,"result")
