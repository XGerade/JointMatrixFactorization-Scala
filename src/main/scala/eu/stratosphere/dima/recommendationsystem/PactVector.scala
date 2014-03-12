/*
 * Project: JointMatrixFactorization
 * @author Xugang Zhou
 * @author Fangzhou Yang
 * @version 1.0
 */

package eu.stratosphere.dima.recommendationsystem

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import eu.stratosphere.types.Value

/*
 * This Pact class is a wrapper of Vector which could be passed between on stratosphere
 */
class PactVector extends Value {
  
  val vectorWritable : VectorWritable = new VectorWritable
  
  /*
   * This method reset the precision of vector
   * which when set to true, it will use float instead of double
   */
  def reset(writesLaxPrecision : Boolean) {
    vectorWritable.setWritesLaxPrecision(writesLaxPrecision);
  }
  
  def set(v: Vector) {
    vectorWritable.set(v)
  }
  
  def get : Vector = {
    val result = vectorWritable.get()
    result
  }
  
  override def read(in: DataInput) {
    vectorWritable.readFields(in)
  }
  
  override def write(out: DataOutput) {
    vectorWritable.write(out)
  }
  
  override def toString : String = {
    val result = vectorWritable.toString()
    result
  }
}
