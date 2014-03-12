/*
 * Project: JointMatrixFactorization
 * @author Xugang Zhou
 * @author Fangzhou Yang
 * @version 1.0
 */

package eu.stratosphere.dima.recommendationsystem;

import eu.stratosphere.api.scala.functions._
import org.apache.mahout.math.Vector;

/*
 * This Join class join each rating entry to a user- or item-feature-vector
 * In order to produce a entry contains the rating value and the feature vector 
 */
class Joint extends JoinFunction[(Int, Int, Float), (Int, PactVector), (Int, Int, Float, PactVector)] {
  /*
   * This override method defines how the join works
   */
  override def apply(l: (Int, Int, Float), r: (Int, PactVector)) : (Int, Int, Float, PactVector) = {
    val userID = l._1
    val itemID = l._2
    val rating = l._3
    val featureVector : Vector = r._2.get
    val vectorWritable : PactVector = new PactVector();
    vectorWritable.set(featureVector)
    val result = (userID, itemID, rating, vectorWritable)
    result
  }
}
