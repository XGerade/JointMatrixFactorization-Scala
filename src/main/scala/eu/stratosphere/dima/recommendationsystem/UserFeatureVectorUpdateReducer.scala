/*
 * Project: JointMatrixFactorization
 * @author Xugang Zhou
 * @author Fangzhou Yang
 * @version 1.0
 */

package eu.stratosphere.dima.recommendationsystem;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import com.google.common.collect.Lists;

import eu.stratosphere.api.scala.functions._
import org.apache.mahout.math.Vector;

/*
 * This Reduce class reduce all entries with same userID to its user-feature-vector
 */
class UserFeatureVectorUpdateReducer extends GroupReduceFunction[(Int, Int, Float, PactVector), (Int, PactVector)] {  
  /*
   * This override method defines how user-feature-vector is calculated from all items' rating and their feature-vector
   * @param in:Iterator[(userID, itemID, rating, user-feature-vector)] List with same userID
   * @return (userID, user-feature-vector)
   */
  override def apply(in: Iterator[(Int, Int, Float, PactVector)]) : (Int, PactVector) = {
    /*
     * Get common variables
     */
    val lambda = Util.lambda
    val numFeatures = Util.numFeatures
    val numItems = Util.numItems
    /*
     * Set vector for put in the item rating from the user
     * The itemID starts from 1
     * So the initialized cardinality would be set to numItems + 1
     */
    val ratingVector: Vector = new RandomAccessSparseVector(numItems + 1, numItems + 1)
    /*
     * Set a Map for all items' feature vectors
     */
    val itemFeatureMatrix = new OpenIntObjectHashMap[Vector](numItems)
     /*
     * Put all items' rating in a vector
     * HashMap all the items' feature-vectors
     */
   var userID = -1
    while (in.hasNext) {
      val temp = in.next()     
      userID = temp._1
      val itemID = temp._2
      val rating = temp._3
      ratingVector.setQuick(itemID, rating)
      val itemFeatureVector : Vector = temp._4.get
      itemFeatureMatrix.put(itemID, itemFeatureVector)
    }
    /*
     * Extract all item-feature-vectors whose itemID rated by the user
     */    
    val featureVectors = Lists.newArrayListWithCapacity[Vector](ratingVector.getNumNondefaultElements())
    val it = ratingVector.nonZeroes().iterator()
    while (it.hasNext) {
      val elem = it.next()
      if (itemFeatureMatrix.containsKey(elem.index)) {
        featureVectors.add(itemFeatureMatrix.get(elem.index))
      }
    }
    /*
     * Calculate the user-feature-vector using Alternative Least Square (ALS) method
     */
    val userFeatureVector : Vector = AlternatingLeastSquaresSolver.solve(featureVectors, ratingVector, lambda, numFeatures)
    val userFeatureVectorWritable = new PactVector()
    userFeatureVectorWritable.set(userFeatureVector)
    val result = (userID, userFeatureVectorWritable)
    result
  }
}