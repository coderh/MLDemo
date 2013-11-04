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

  def E(u: Double, v: Double) = pow(u * exp(v) - 2 * v * exp(-u), 2)

  def partial_E_u(u: Double, v: Double) = 2 * (u * exp(v) - 2 * v * exp(-u)) * (exp(v) + 2 * v * exp(-u))

  def partial_E_v(u: Double, v: Double) = 2 * (u * exp(v) - 2 * v * exp(-u)) * (u * exp(v) - 2 * exp(-u))

  val eta = 0.1

  var w = DenseVector[Double](1, 1)

  def gradDesc = {
    for (i <- 1 to 10) {
      val gradient = DenseVector[Double](partial_E_u(w(0), w(1)), partial_E_v(w(0), w(1)))
      w = w - gradient * eta
    }
  }

  def coordGradDesc = {
    for (i <- 1 to 15) {
      val coordGrad_u = DenseVector[Double](partial_E_u(w(0), w(1)), 0)
      w = w - coordGrad_u * eta
      val coordGrad_v = DenseVector[Double](0, partial_E_v(w(0), w(1)))
      w = w - coordGrad_v * eta
    }
  }

  def main(args: Array[String]) = {
    //    coordGradDesc
    gradDesc
    println(w)
    println(E(w(0), w(1)))
  }
}
