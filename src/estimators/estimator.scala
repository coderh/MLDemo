package estimators

import scala.math.min
import scala.math.abs
import scala.util.Random

/**
 * Created with IntelliJ IDEA.
 * User: coderh
 * Date: 11/15/13
 * Time: 3:02 PM
 */
object estimator {

  def run() = {
    val n = 1000000

    val l1 = 1 to n map (_ => Random.nextDouble)
    val exp1 = l1.sum / n

    val l2 = 1 to n map (_ => Random.nextDouble)
    val exp2 = l2.sum / n

    val lmin = l1 zip l2 map (p => min(p._1, p._2))
    val expMin = lmin.sum / n

    def close(a: Double, x1: Double, x2: Double) = {
      if (abs(a - x1) < abs(a - x2)) x1 else x2
    }

    //l1 zip l2 foreach (p => println("(" + p._1 + ", " + p._2 + ") => " + min(p._1, p._2)))

    println("E(e1) = " + exp1)
    println("E(e2) = " + exp2)
    println("E(min(e1, e1)) = " + expMin)
    println("close to: " + close(expMin, 0.25, 0.4))
  }

  def main(args: Array[String]) = {
    run()
  }

}
