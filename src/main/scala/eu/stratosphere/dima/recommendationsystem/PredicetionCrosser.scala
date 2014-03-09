package eu.stratosphere.dima.recommendationsystem;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import eu.stratosphere.api.scala.functions._

class PredicetionCrosser extends CrossFunction[(Int, PactVector), (Int, PactVector), (Int, Int, Float)]{
  override def apply (l: (Int, PactVector), r: (Int, PactVector)) : (Int, Int, Float) = {
    val userID = r._1
    val itemID = l._1
    val itemVector = l._2.get
    val userVector = r._2.get
    val predictRating : Float = itemVector.dot(userVector).toFloat
    (userID, itemID, predictRating)
  }
}
