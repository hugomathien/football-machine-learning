from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.feature import StandardScalerModel
# $example off$

from pyspark import SparkContext

if __name__ == "__main__":
    sc = SparkContext(appName="Regression Metrics Example")

# $example on$
# Load and parse the data
def parsePoint(line):
    values = line.split()
    y = int(values[0])
    x = [int(x.split(':')[1]) for x in values[1:]]
    #xSquare = [a*a for a in x]
    #xCube = [a*a*a for a in x]
    #x = x + xSquare
    return LabeledPoint(y,x)

data = sc.textFile("/Users/hugomathien/Documents/workspace/footballdata/learning_vector/learningVector_extended.txt")
parsedData = data.map(parsePoint)

splits = parsedData.randomSplit([0.6, 0.4],1234)
train = splits[0]
test = splits[1]
# Build the model
#model = SVMWithSGD.train(train, iterations=10000,step=0.001) 0.3712
model = SVMWithSGD.train(train, iterations=10000,step=0.001)
# Evaluating the model on training data
labelsAndPredsTrain = train.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPredsTrain.filter(lambda (v, p): v != p).count() / float(train.count())
labelsAndPredsTest = test.map(lambda p: (p.label, model.predict(p.features)))
testErr = labelsAndPredsTest.filter(lambda (v, p): v != p).count() / float(test.count())
print("Training Set size = " + str(float(train.count())))
print("Training Error = " + str(trainErr))
print("Test Set size = " + str(float(test.count())))
print("Test Error = " + str(testErr))

# Save and load model
#model.save(sc, "myModelPath")
#sameModel = SVMModel.load(sc, "myModelPath")