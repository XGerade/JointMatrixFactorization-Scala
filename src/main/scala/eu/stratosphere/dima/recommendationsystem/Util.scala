/*
 * Project: JointMatrixFactorization
 * @author Xugang Zhou
 * @author Fangzhou Yang
 * @version 1.0
 */

package eu.stratosphere.dima.recommendationsystem

/*
 * This object defines common variables for in this project
 */
object Util {
  /*
   * Initial values are set in the case from movieLens-100k
   */
  var numFeatures = 10
  var lambda = 0.1
  var numUsers = 943
  var numItems = 1682
  var maxUserID = 943
  var maxItemID = 1682
  
  /*
   * The method is defined to reset all the parameters
   */
  def setParameters(newLambda: Float, newNumFeatures: Int, newNumUsers: Int, newNumItems: Int) {
    lambda = newLambda
    numFeatures = newNumFeatures
    numUsers = newNumUsers
    numItems = newNumItems
    maxUserID = newNumUsers
    maxItemID = newNumItems
  }
}