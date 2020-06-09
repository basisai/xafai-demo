import random
import numpy as np
from pyspark.sql import Row
from sklearn import neighbors
from pyspark.ml.feature import VectorAssembler


def vectorizer_func(dataInput, TargetFieldName):
    if(dataInput.select(TargetFieldName).distinct().count() != 2):
        raise ValueError("Target field must have only 2 distinct classes")
        
    columnNames = list(dataInput.columns)
    columnNames.remove(TargetFieldName)
    
    dataInput = dataInput.select((','.join(columnNames)+','+TargetFieldName).split(','))
    
    assembler=VectorAssembler(inputCols=columnNames, outputCol='features')
    pos_vectorized = assembler.transform(dataInput)
    vectorized = (
        pos_vectorized
        .select('features', TargetFieldName)
        .withColumn('label', pos_vectorized[TargetFieldName]).
        drop(TargetFieldName)
    )
    return vectorized


def smote_sampling(vectorized, k=5, minority_class=1, majority_class=0, pct_over=200, pct_under=100):
    if(pct_under > 100 | pct_under < 10):
        raise ValueError("Percentage Under must be in range 10 - 100")
        
    if(pct_over < 100):
        raise ValueError("Percentage Over must be in at least 100");
        
    dataInput_min = vectorized[vectorized['label'] == minority_class]
    dataInput_maj = vectorized[vectorized['label'] == majority_class]
    
    feature = (
        dataInput_min
        .select('features')
        .rdd
        .map(lambda x: x[0])
    ).collect()
    feature = np.asarray(feature)
    
    nbrs = neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto').fit(feature)
    neighbours =  nbrs.kneighbors(feature)
    gap = neighbours[0]
    neighbours = neighbours[1]
    
    min_rdd = dataInput_min.drop('label').rdd
    pos_rddArray = min_rdd.map(lambda x: list(x))
    pos_ListArray = pos_rddArray.collect()
    min_Array = list(pos_ListArray)
    
    newRows = []
    nt = len(min_Array)
    nexs = pct_over/100
    for i in range(nt):
        for j in range(nexs):
            neigh = random.randint(1,k)
            difs = min_Array[neigh][0] - min_Array[i][0]
            newRec = (min_Array[i][0]+random.random()*difs)
            newRows.insert(0,(newRec))
            
    newData_rdd = sc.parallelize(newRows)
    newData_rdd_new = newData_rdd.map(lambda x: Row(features = x, label = 1))
    new_data = newData_rdd_new.toDF()
    new_data_minor = dataInput_min.unionAll(new_data)
    new_data_major = dataInput_maj.sample(False, (float(pct_under)/float(100)))
    return new_data_major.unionAll(new_data_minor)