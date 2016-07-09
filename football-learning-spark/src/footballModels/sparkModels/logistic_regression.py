#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Logistic regression using MLlib.

This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function

import sys
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD


def parsePoint(line):
    """
    Parse a line of text into an MLlib LabeledPoint object.
    """
    values = [s for s in line.split(' ')]
    y = int(values[0])
    a = values[1:]
    x = [int(s.split(':')[1]) for s in a]
    #b = [item for sublist in a for item in sublist]
    #str = ''.join(b)
    #c = [s for s in str.split(':')]
    if values[0] == -1:   # Convert -1 labels to 0 for MLlib
        values[0] = 0
    return LabeledPoint(y, x)


if __name__ == "__main__":
    
    sc = SparkContext(appName="PythonLR")
    sqlContext = SQLContext(sc)
    data = sc.textFile("/Users/hugomathien/Documents/workspace/footballdata/learning_vector/learningVector_extended.txt")
    points = data.map(parsePoint)
    iterations = int(100000)
    model = LogisticRegressionWithSGD.train(points, iterations,step=0.000000001)
    labelsAndPreds = points.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(points.count())
    print("Training Error = " + str(trainErr))
    print("Final weights: " + str(model.weights))
    print("Final intercept: " + str(model.intercept))
    
    
    sc.stop()
