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
 * This Reduce class reduce all entries with same itemID to its item-feature-vector
 */
class ItemFeatureVectorUpdateReducer extends GroupReduceFunction[(Int, Int, Float, PactVector), (Int, PactVector)] {
  /*
   * This override method defines how item-feature-vector is calculated from all users' rating and their feature-vector
   * @param in:Iterator[(userID, itemID, rating, user-feature-vector)] List with same itemID
   * @return (itemID, item-feature-vector)
   */
  override def apply (in: Iterator[(Int, Int, Float, PactVector)]) : (Int, PactVector) = {
    /*
     * Get common variables
     */
    val lambda = Util.lambda
    val numFeatures = Util.numFeatures
    val numUsers = Util.numUsers
    val ratingVector: Vector = new RandomAccessSparseVector(Integer.MAX_VALUE, numUsers + 1)
    /*
     * Set a Map for all users' feature vectors
     */
    val userFeatureMatrix = new OpenIntObjectHashMap[Vector](numUsers)
    /*
     * Put all users' rating in a vector
     * HashMap all the users' feature-vectors
     */
    var itemID = -1
    while (in.hasNext) {
      val temp = in.next()
      val userID = temp._1
      itemID = temp._2
      val rating = temp._3
      ratingVector.setQuick(userID, rating)
      val userFeatureVector : Vector = temp._4.get
      userFeatureMatrix.put(userID, userFeatureVector)
    }
    /*
     * Extract all user-feature-vectors whose userID rated for the item
     */
    val featureVectors = Lists.newArrayListWithCapacity[Vector](ratingVector.getNumNondefaultElements())
    val itemRatingVector : Vector = new SequentialAccessSparseVector(ratingVector)
    val it = itemRatingVector.nonZeroes().iterator()
    while (it.hasNext) {
      val elem = it.next()
      if (userFeatureMatrix.containsKey(elem.index)) {
        featureVectors.add(userFeatureMatrix.get(elem.index))
      }
    }
    /*
     * Calculate the item-feature-vector using Alternative Least Square (ALS) method
     */
    val itemFeatureVector : Vector = AlternatingLeastSquaresSolver.solve(featureVectors, itemRatingVector, lambda, numFeatures)
    val itemFeatureVectorWritable = new PactVector()
    itemFeatureVectorWritable.set(itemFeatureVector)
    val result = (itemID, itemFeatureVectorWritable)
    result

  }  
}
