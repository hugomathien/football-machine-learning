'''
Created on Jun 29, 2016

@author: hugomathien
'''
# Imports
# Take care about unused imports (and also unused variables),
# please comment them all, otherwise you will get any errors at the execution.
# Note that neither the directives "@PydevCodeAnalysisIgnore" nor "@UnusedImport"
# will be able to solve that issue.
#from pyspark.mllib.clustering import KMeans
from pyspark import SparkConf, SparkContext
import os

# Configure the Spark environment
sparkConf = SparkConf().setAppName("WordCounts").setMaster("local")
sc = SparkContext(conf = sparkConf)

 # The WordCounts Spark program
textFile = sc.textFile(os.environ["SPARK_HOME"] + "/README.md")
wordCounts = textFile.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
for wc in wordCounts.collect(): print wc