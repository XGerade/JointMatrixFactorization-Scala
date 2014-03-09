package eu.stratosphere.dima.recommendationsystem;


import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import com.google.common.collect.Lists;

import eu.stratosphere.api.scala.functions._
import org.apache.mahout.math.Vector;

class UserFeatureVectorUpdateReducer extends GroupReduceFunction[(Int, Int, Float, PactVector), (Int, PactVector)] {
  
  override def apply(in: Iterator[(Int, Int, Float, PactVector)]) : (Int, PactVector) = {
    
    val lambda = Util.lambda
    val numFeatures = Util.numFeatures
    val numItems = Util.numItems
    val userFeatureVectorWritable = new PactVector()
    val vector: Vector = new RandomAccessSparseVector(Integer.MAX_VALUE, numItems)
    val itemFeatureMatrix = new OpenIntObjectHashMap[Vector](numItems)
    var userID = -1
    while (in.hasNext) {
      val temp = in.next()
      
      userID = temp._1
      val itemID = temp._2
      val rating = temp._3
      
      vector.setQuick(itemID, rating)
      
      val itemFeatureVector : Vector = temp._4.get
      
      itemFeatureMatrix.put(itemID, itemFeatureVector)
    }
    
    val userRatingVector : Vector = new SequentialAccessSparseVector(vector)
    
    val featureVectors = Lists.newArrayListWithCapacity[Vector](userRatingVector.getNumNondefaultElements())
    
    val it = userRatingVector.nonZeroes.iterator

    while (it.hasNext) {
      val elem = it.next()
      if (itemFeatureMatrix.containsKey(elem.index)) {
        featureVectors.add(itemFeatureMatrix.get(elem.index))
      }
    }
    
    val userFeatureVector : Vector = AlternatingLeastSquaresSolver.solve(featureVectors, userRatingVector, lambda, numFeatures)
    userFeatureVectorWritable.set(userFeatureVector)
    val result = (userID, userFeatureVectorWritable)
    result
  }
}