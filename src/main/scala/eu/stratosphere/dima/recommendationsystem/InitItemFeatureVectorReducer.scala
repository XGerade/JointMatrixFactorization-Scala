package eu.stratosphere.dima.recommendationsystem;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import eu.stratosphere.api.scala.functions._
import java.util.Random;

class InitItemFeatureVectorReducer extends GroupReduceFunction[(Int, Int, Float), (Int, PactVector)] {
	override def apply (in: Iterator[(Int, Int, Float)]) : (Int, PactVector) = {
	  val numfeatures = Util.numFeatures
	  val random = new Random()
	  val featureVector = new PactVector()
	  val features : Vector = new SequentialAccessSparseVector(Integer.MAX_VALUE, Util.numFeatures)
	  val itemID = in.next._2;
	  for (i <- 1 to numfeatures) {
	    val t = random.nextFloat
	    features.setQuick(i - 1, t)
	  }
	  featureVector.set(features)
	  (itemID, featureVector)
	}
}

