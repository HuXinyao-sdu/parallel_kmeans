from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans #导入MLlib的KMeans算法
from pyspark.ml.feature import VectorAssembler #导入特征组装器，将多个数值列拼成一个向量列
from pyspark.ml.linalg import Vectors #导入向量工具类 

def main():
	#构建SparkSession实例
	spark=SparkSession.builder.appName("KMeansMLlib").master("local[*]").getOrCreate()
	
	#读取csv二维点表
	df=spark.read.csv("../data/points.csv",header=False,inferSchema=True)
	#将默认列名(_c0,_c1)改为(x,y)
	df=df.withColumnRenamed("_c0","x").withColumnRenamed("_c1","y")
	
	#声明特征组装器实现x,y(two cols)->features(one col)
	assembler=VectorAssembler(inputCols=["x","y"],outputCol="features")
	#声明新列：features:[x,y]
	df=assembler.transform(df)
	#转换后df:x(x),y(y),features([x,y])
	
	#初始化KMeans模型
	kmeans=KMeans(k=3,seed=42,maxIter=100,tol=1e-6,featuresCol="features")
	model=kmeans.fit(df) #训练模型
	centers=model.clusterCenters() #获取最终聚类中心并打印
	print("MLlib中心：",centers)
	
	#关闭sparkSession实例，释放资源
	spark.stop()

if __name__=="__main__":
	main()	
