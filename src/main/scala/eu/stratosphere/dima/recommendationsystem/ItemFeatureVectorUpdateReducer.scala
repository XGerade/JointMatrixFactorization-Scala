package eu.stratosphere.dima.recommendationsystem;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import com.google.common.collect.Lists;

import eu.stratosphere.api.scala.functions._
import org.apache.mahout.math.Vector;

class ItemFeatureVectorUpdateReducer extends GroupReduceFunction[(Int, Int, Float, PactVector), (Int, PactVector)] {
  
  override def apply (in: Iterator[(Int, Int, Float, PactVector)]) : (Int, PactVector) = {
    val lambda = Util.lambda
    val numFeatures = Util.numFeatures
    val numUsers = Util.numUsers
    val itemFeatureVectorWritable = new PactVector()
    val vector: Vector = new RandomAccessSparseVector(Integer.MAX_VALUE, numUsers)
    val userFeatureMatrix = new OpenIntObjectHashMap[Vector](numUsers)
    var itemID = -1
    while (in.hasNext) {
      val temp = in.next()
      
      val userID = temp._1
      itemID = temp._2
      val rating = temp._3
      
      vector.setQuick(userID, rating)
      
      val userFeatureVector : Vector = temp._4.get
      
      userFeatureMatrix.put(userID, userFeatureVector)
    }
    
    val itemRatingVector : Vector = new SequentialAccessSparseVector(vector)
   
    val featureVectors = Lists.newArrayListWithCapacity[Vector](itemRatingVector.getNumNondefaultElements())
    
    val it = itemRatingVector.nonZeroes.iterator

    while (it.hasNext) {
      val elem = it.next()
      if (userFeatureMatrix.containsKey(elem.index)) {
        featureVectors.add(userFeatureMatrix.get(elem.index))
      }
    }
    
    val itemFeatureVector : Vector = AlternatingLeastSquaresSolver.solve(featureVectors, itemRatingVector, lambda, numFeatures)
    itemFeatureVectorWritable.set(itemFeatureVector)
    val result = (itemID, itemFeatureVectorWritable)
    result

  }  
}
