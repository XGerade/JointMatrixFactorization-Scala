/*
 * Project: JointMatrixFactorization
 * @author Xugang Zhou
 * @author Fangzhou Yang
 * @version 1.0
 */

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

/*
 * This Class is the the "plan" class of this project.
 */
class JointMatrixFactorization extends Program with ProgramDescription with Serializable {
  override def getDescription() = {
    "Parameters: [numSubStasks] [input] [output] [lambda] [numFeatures] [numUsers] [numItems] [numIter]"
  }
  override def getPlan(args: String*) = {
    getScalaPlan(args(0).toInt, args(1), args(2), args(3).toFloat, args(4).toInt, args(5).toInt, args(6).toInt, args(7).toInt)
  }

  /*
   * This method defines how the data would be operated.
   * @return The whole scala-plan
   * @param args(0) Number of subtasks to specify parallelism
   * @param args(1) Path to input file
   * @param args(2) Path to output file
   * @param args(3) lambda which is used during ALS learning
   * @param args(4) Number of features pre-set for learning
   * @param args(5) Number of Users of input
   * @param args(6) Number of Items of input
   * @param args(7) Number of Iterations to run in ALS 
   */
  def getScalaPlan(numSubTasks:Int, inputPath: String, outputPath: String, lambda: Float, numFeatures: Int, numUsers: Int, numItems: Int, numIter: Int) = {

    println("Job Started")
    println("InputPath: " + inputPath)
    println("OutputPath: " + outputPath)

    Util.setParameters(lambda, numFeatures, numUsers, numItems)

    val tupple = DataSource(inputPath, CsvInputFormat[(Int, Int, Float)](Seq(0, 1, 2), "\n", '\t'))
    
    /*
     * Initialize item-feature-vectors with random value
     */
    val initItemFeatureVector = tupple groupBy { x => x._2} reduceGroup { new InitItemFeatureVectorReducer }    
    	 
    /*
     * Learn the user-feature-vectors with initialized item-feature-vectors
     */
    var userFeatureVectorJoin = tupple join initItemFeatureVector where {case (a, b, c) => b} isEqualTo {case (a, b) => a} map {new Joint} 
    var userFeatureVectorReduce = userFeatureVectorJoin groupBy {case (userID, _, _, _) => userID} reduceGroup {new UserFeatureVectorUpdateReducer}
    
    /*
     * Learn the item-feature-vectors with user-feature-vectors
     */
    var itemFeatureVectorJoin = tupple join userFeatureVectorReduce where {case (a, b, c) => a} isEqualTo {case (a, b) => a} map {new Joint}
    var itemFeatureVectorReduce = itemFeatureVectorJoin groupBy {case (_, itemID, _, _) => itemID} reduceGroup {new ItemFeatureVectorUpdateReducer}
    
    /*
     * Continue to Alternative-Least-Sqaure (ALS) learning with numIter iterations
     */
    val numIterations = numIter
    for (i <- 1 to numIterations) {
      userFeatureVectorJoin = tupple join itemFeatureVectorReduce where {case (a, b, c) => b} isEqualTo {case (a, b) => a} map {new Joint} 
      userFeatureVectorReduce = userFeatureVectorJoin groupBy {case (userID, _, _, _) => userID} reduceGroup {new UserFeatureVectorUpdateReducer}
      itemFeatureVectorJoin = tupple join userFeatureVectorReduce where {case (a, b, c) => a} isEqualTo {case (a, b) => a} map {new Joint}
      itemFeatureVectorReduce = itemFeatureVectorJoin groupBy {case (_, itemID, _, _) => itemID} reduceGroup {new ItemFeatureVectorUpdateReducer}
    }
    
    /*
     * Use learned user- and item-feature-vectors to do the prediction of rating
     */
    val prediction = itemFeatureVectorReduce cross userFeatureVectorReduce map {new PredicetionCrosser}
    
    /*
     * Put the predicted-rating result to output stream
     */
    val output = prediction.write(outputPath, CsvOutputFormat("\n", "\t"))
  
    /*
     * Return the plan
     */
    val plan = new ScalaPlan(Seq(output), "Rating Prediction Computation")
    plan.setDefaultParallelism(numSubTasks)
    plan
  }
}

/*
 * This object enables you to run this project locally.
 * Run this object with the parameters specified below will result in run this project locally.
 */
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
