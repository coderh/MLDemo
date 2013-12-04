package generalization_error

import scala.math._

/**
 *
 * Author: coderh
 * Date: 10/27/13
 * Time: 11:08 PM
 *
 */
object genError {

  val d_vc: Double = 50

  val delta: Double = 0.05

  def m_H(N: Double) = pow(N, d_vc)

  def quadraticEquationSolver(a: Double, b: Double, c: Double) = (-b + sqrt(b * b - 4 * a * c)) / (2 * a)

  // VCBound
  def bound_1(N: Double) = sqrt(8 / N * log(4 * m_H(2 * N) / delta))

  // Rademacher Penalty Bound
  def bound_2(N: Double) = sqrt(2 * log(2 * N * m_H(N)) / N) + sqrt(2 * log(1 / delta) / N) + 1 / N

  // Parrondo and Van den Broek Bound
  def bound_3(N: Double) = quadraticEquationSolver(1, -2 / N, -log(6 * m_H(2 * N) / delta) / N)

  // Devroye Bound
  def bound_4(N: Double) = quadraticEquationSolver(2 * N - 4, -4, -log(4 * m_H(N * N) / delta))

  //  def bound_4(N: Double) = quadraticEquationSolver(2 * N - 4, -4, -(log10(4) + 400) / log10(E) + log(delta))

  def main(args: Array[String]) {
    val N = 5
    println(bound_1(N))
    println(bound_2(N))
    println(bound_3(N))
    println(bound_4(N))

    val ns = List(400000, 420000, 440000, 460000, 480000)
    def diff(N: Int) = sqrt(8 / N.toDouble * log(4 * pow(2 * N, 10) / 0.05)) - 0.05
    println(ns.map(diff).mkString("\n"))

    // TODO: plot curve
  }


}
