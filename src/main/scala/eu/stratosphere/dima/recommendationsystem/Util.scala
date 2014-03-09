package eu.stratosphere.dima.recommendationsystem;

object Util {
  var numFeatures = 10
  var lambda = 0.1
  var numUsers = 943
  var numItems = 1682
  var maxUserID = 943
  var maxItemID = 1682

  def setParameters(newLambda: Float, newNumFeatures: Int, newNumUsers: Int, newNumItems: Int) {
    lambda = newLambda
    numFeatures = newNumFeatures
    numUsers = newNumUsers
    numItems = newNumItems
    maxUserID = newNumUsers
    maxItemID = newNumItems
  }
}