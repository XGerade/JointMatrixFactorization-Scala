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

import eu.stratosphere.api.scala.functions._
import java.util.Random;

/*
 * This Reduce class reduce all the input entries to each itemID and a random feature-vector 
 */
class InitItemFeatureVectorReducer extends GroupReduceFunction[(Int, Int, Float), (Int, PactVector)] {
  /*
   * This override method defines how the entries with same itemID reduce to a random feature-vector
   * @param in:Iterator[(userID, itemID, rating)] List of entries with same itemID
   * @return (itemID, item-feature-vector)
   */
  override def apply (in: Iterator[(Int, Int, Float)]) : (Int, PactVector) = {
    /*
     * Get number of features and set a vector for it
     */
    val numfeatures = Util.numFeatures
    val features : Vector = new SequentialAccessSparseVector(Integer.MAX_VALUE, Util.numFeatures)

    val random = new Random()
    val featureVector = new PactVector()
    val itemID = in.next._2;
    /*
     * Give each feature a random value
     */
    for (i <- 1 to numfeatures) {
      val t = random.nextFloat
      features.setQuick(i - 1, t)
    }
    
    featureVector.set(features)
    (itemID, featureVector)
  }
}

