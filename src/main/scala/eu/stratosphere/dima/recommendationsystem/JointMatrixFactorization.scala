package eu.stratosphere.dima.recommendationsystem;

import java.util.Random;

import com.google.common.primitives.Longs

import org.apache.mahout.math.Vector
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.VectorWritable

import eu.stratosphere.api.common.Program
import eu.stratosphere.api.common.ProgramDescription
import eu.stratosphere.api.scala.functions._

import eu.stratosphere.client.LocalExecutor
import eu.stratosphere.api.scala.TextFile
import eu.stratosphere.api.scala.ScalaPlan
import eu.stratosphere.api.scala._
import eu.stratosphere.api.scala.operators._
import eu.stratosphere.client.RemoteExecutor

object RunJointMatrixFactorizationLocal {
  def main(args: Array[String]) {
    val job = new JointMatrixFactorization
    if (args.size < 7) {
      println(job.getDescription)
      return
    }
    val plan = job.getScalaPlan(args(0).toInt, args(1), args(2), args(3).toFloat, args(4).toInt, args(5).toInt, args(6).toInt, args(7).toInt)
    LocalExecutor.execute(plan)
    System.exit(0)
  }
}

class JointMatrixFactorization extends Program with ProgramDescription with Serializable {
  override def getDescription() = {
    "Parameters: [numSubStasks] [input] [output] [lambda] [numFeatures] [numUsers] [numItems] [numIter]"
  }
  override def getPlan(args: String*) = {
    getScalaPlan(args(0).toInt, args(1), args(2), args(3).toFloat, args(4).toInt, args(5).toInt, args(6).toInt, args(7).toInt)
  }


  
  def getScalaPlan(numSubTasks:Int, inputPath: String, outputPath: String, lambda: Float, numFeatures: Int, numUsers: Int, numItems: Int, numIter: Int) = {
    println("Job Started")
    println("InputPath: " + inputPath)
    println("OutputPath: " + outputPath)
    Util.setParameters(lambda, numFeatures, numUsers, numItems)

    val tupple = DataSource(inputPath, CsvInputFormat[(Int, Int, Float)](Seq(0, 1, 2), "\n", '\t'))
    
    val initItemFeatureVector = tupple groupBy { x => x._2} reduceGroup { new InitItemFeatureVectorReducer }    
    	 
    var userFeatureVectorJoin = tupple join initItemFeatureVector where {case (a, b, c) => b} isEqualTo {case (a, b) => a} map {new Joint} 
    var userFeatureVectorReduce = userFeatureVectorJoin groupBy {case (userID, _, _, _) => userID} reduceGroup {new UserFeatureVectorUpdateReducer}
    
    var itemFeatureVectorJoin = userFeatureVectorJoin
    var itemFeatureVectorReduce = userFeatureVectorReduce
    
    println("Iteration Started")
    val numIterations = numIter
    for (i <- 1 to numIterations) {
      itemFeatureVectorJoin = tupple join userFeatureVectorReduce where {case (a, b, c) => a} isEqualTo {case (a, b) => a} map {new Joint}
      itemFeatureVectorReduce = itemFeatureVectorJoin groupBy {case (_, itemID, _, _) => itemID} reduceGroup {new ItemFeatureVectorUpdateReducer}
 
      userFeatureVectorJoin = tupple join itemFeatureVectorReduce where {case (a, b, c) => b} isEqualTo {case (a, b) => a} map {new Joint} 
      userFeatureVectorReduce = userFeatureVectorJoin groupBy {case (userID, _, _, _) => userID} reduceGroup {new UserFeatureVectorUpdateReducer}
    }

    itemFeatureVectorJoin = tupple join userFeatureVectorReduce where {case (a, b, c) => a} isEqualTo {case (a, b) => a} map {new Joint}
    itemFeatureVectorReduce = itemFeatureVectorJoin groupBy {case (_, itemID, _, _) => itemID} reduceGroup {new ItemFeatureVectorUpdateReducer}
    
    val prediction = itemFeatureVectorReduce cross userFeatureVectorReduce map {new PredicetionCrosser}
    
    val output = prediction.write(outputPath, CsvOutputFormat("\n", "\t"))
  
    val plan = new ScalaPlan(Seq(output), "Rating Prediction Computation")
    plan.setDefaultParallelism(numSubTasks)
    plan
  }
}
