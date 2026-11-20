from pyspark import SparkContext,SparkConf
import numpy as np

#解析行内容
def parse_line(line):
	parts=line.strip().split(",")
	#返回二维点的(x,y)坐标
	return np.array([float(parts[0]),float(parts[1])])

def dist_sq(point,centroids):
	#返回点到各中心的平方距离数组 
	return np.sum((centroids-point)**2,axis=1)

def assign_label(point,centroids):
	d=dist_sq(point,centroids)
	return int(np.argmin(d)) #返回最近距离的簇索引

def main():
	conf=SparkConf().setAppName("ParallelKMeans").setMaster("local[*]")
	sc=SparkContext(conf=conf)
	
	#1.读取数据
	lines=sc.textFile("../data/points.csv")
	data=lines.map(parse_line).cache()
	n=data.count() #统计总二维点数量
	
	#2.初始化中心
	all_points=data.collect() #将二维点数组去重
	rng=np.random.RandomState(42)
	idx=rng.choice(n,size=3,replace=False)
	centroids=np.array([all_points[i] for i in idx])
	
	print("初始中心:\n",centroids)
	
	k=3
	max_iter=100
	tol=1e-6
	
	for it in range(max_iter):
		#3.广播中心
		bc=sc.broadcast(centroids)
		
		#4.给每个点分配标签，output(label,point)
		labeled=data.map(lambda p : (assign_label(p,bc.value),p)).cache()
		
		#5.聚合求新中心
		#a[0],b[0]:(x1,y1),(x2,y2)
		#a[1],b[1]:1,1
		agg=labeled.mapValues(lambda p :(p,1)).reduceByKey(lambda a,b:(a[0]+b[0],a[1]+b[1])).collect()
		
		new_centroids=centroids.copy()
		for label,(s,c) in agg:
		#s:sum of (x,y):(x_sum,y_sum)
		#c:count of (x,y)
			new_centroids[label]=s/c #计算平均(x_mean,y_mean)
			
		shift=np.sum((new_centroids-centroids)**2) #计算簇中心变化情况
		print(f"[iter {it+1}] shift={shift:.8f}")
		centroids=new_centroids #更新簇中心
		labeled.unpersist()
		if shift < tol: #簇中心基本不变
			break #提前终止循环
	#6.输出最终迭代结果:（labels,point(x,y)）
	bc=sc.broadcast(centroids) #再次广播计算
	final=data.map(lambda p : (assign_label(p,bc.value),p)).collect() #计算簇结果并去重
	
	#7.按中心坐标排序并进行标签重映射
	#centroids[:,1],centroids[:,0]:
	order=np.lexsort((centroids[:,1],centroids[:,0])) #新排序索引
	sorted_centroids=centroids[order]  #排序后聚类中心
	mapping={int(old):int(new) for new,old in enumerate(order)} #建立新旧排序索引映射关系
	
	#zipWithIndex():给每个元素配置一个递增的0起始编号，x->(x,i)
	#.map(lambda x : (x[1],x[0]))：即(x,i)->(i,x)
	indexed=data.zipWithIndex().map(lambda x :(x[1],x[0]))
	#labeled_indexed:(i,x)->(i,clusterID)
	labeled_indexed=indexed.map(lambda x : (x[0],assign_label(x[1],bc.value)))
	#pairs:(i,clusterID)->clusterIDValue
	pairs=labeled_indexed.sortByKey().map(lambda x :mapping[x[1]]).collect()
	#保存聚类中心和标签
	np.savetxt("result_centroids_rdd.csv",
				sorted_centroids,delimiter=",",fmt="%.6f")
	np.savetxt("result_labels_rdd.csv",np.array(pairs),fmt="%d")
	print("并行结果已保存至result_centroids_rdd.csv result_labels_rdd.csv")
	
	sc.stop()

if __name__ == "__main__":
	main()
	
