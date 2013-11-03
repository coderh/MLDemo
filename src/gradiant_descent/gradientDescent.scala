package gradiant_descent

import breeze.linalg._
import scala.math.{exp, pow}

/**
 *
 * Author: coderh
 * Date: 11/4/13
 * Time: 12:08 AM
 *
 */

object gradientDescent {

  val eta = 0.1

  def E(u: Double, v: Double) = pow(u * exp(v) - 2 * v * exp(-u), 2)

  def partial_E_u(u: Double, v: Double) = 2 * (u * exp(v) - 2 * v * exp(-u)) * (exp(v) + 2 * v * exp(-u))

  def partial_E_v(u: Double, v: Double) = 2 * (u * exp(v) - 2 * v * exp(-u)) * (u * exp(v) - 2 * exp(-u))

  def main(args: Array[String]) = {
    var w = DenseVector[Double](1, 1)

    for(i <- 1 to 10) {
      val gradient = DenseVector[Double](partial_E_u(w(0), w(1)), partial_E_v(w(0), w(1)))
      w -= gradient * eta
    }

    println(w)
    println(E(w(0), w(1)))
  }
}
