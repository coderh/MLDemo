package bias_and_variance

import scala.math._
import breeze.plot._
import scala.util.Random
import breeze.linalg._

/**
 *
 * Author: coderh
 * Date: 10/28/13
 * Time: 12:13 AM
 *
 */
object biasVariance {

  case class DataPoint(x: Double, y: Double)

  def func(x: Double) = sin(Pi * x)

  def integration(f: Double => Double, start: Double, end: Double, stepNb: Int = 1000) = {
    val step = (end - start) / stepNb.toDouble
    val xs = Array.tabulate(stepNb + 1)(n => start + step * n)
    xs.map(x => step * f(x)).sum
  }

  def generate2Points = {
    val x1 = 2 * Random.nextDouble - 1
    val x2 = 2 * Random.nextDouble - 1
    (DataPoint(x1, func(x1)), DataPoint(x2, func(x2)))
  }

  def getIntercept(pts: (DataPoint, DataPoint)) = {
    pts._1.y + pts._2.y / 2
  }

  def getSlope(pts: (DataPoint, DataPoint)) = {
    val ptList = List(List(pts._1.x), List(pts._2.x))
    val X = DenseVector.tabulate(2)(i => ptList(i)(0))
    val Y = DenseVector(pts._1.y, pts._2.y)
    val w = 1 / X.dot(X) * X.dot(Y)
    w
  }

  def getLine(pts: (DataPoint, DataPoint)) = {
    val ptList = List(List(1.toDouble, pts._1.x), List(1.toDouble, pts._2.x))
    val X = DenseMatrix.tabulate(2, 2)((i: Int, j: Int) => ptList(i)(j))
    val Y = DenseVector(pts._1.y, pts._2.y)
    val Xt = X.t
    val w = inv(Xt * X) * Xt * Y
    w
  }

  def getQuadratic(pts: (DataPoint, DataPoint)) = {
    val ptList = List(List(pts._1.x * pts._1.x), List(1.toDouble, pts._2.x * pts._2.x))
    val X = DenseVector.tabulate(2)(i => ptList(i)(0))
    val Y = DenseVector(pts._1.y, pts._2.y)
    val w = 1 / X.dot(X) * X.dot(Y)
    w
  }

  def getQuadraticWithIntercept(pts: (DataPoint, DataPoint)) = {
    val ptList = List(List(1.toDouble, pts._1.x * pts._1.x), List(1.toDouble, pts._2.x * pts._2.x))
    val X = DenseMatrix.tabulate(2, 2)((i: Int, j: Int) => ptList(i)(j))
    val Y = DenseVector(pts._1.y, pts._2.y)
    val Xt = X.t
    val w = inv(Xt * X) * Xt * Y
    w
  }

  def main(args: Array[String]) {

    val setNb = 10000
    val dataSets = List.tabulate(setNb)(_ => generate2Points)

    // h(x) = b
    val bs = dataSets.map(getIntercept)
    val b_bar = bs.sum / setNb

    // h(x) = ax
    val as = dataSets.map(getSlope)
    val a_bar = as.sum / setNb

    // h(x) = ax +b
    val lines = dataSets.map(getLine)
    val line_bar = lines.reduce(_ + _).map(_ / setNb)

    // h(x) = ax^2
    val quadList = dataSets.map(getQuadratic)
    val quad_bar = quadList.sum / setNb

    // h(x) = ax^2 + b

    val quadWithInterceptList = dataSets.map(getQuadraticWithIntercept)
    val quadWithIntercept_bar = quadWithInterceptList.reduce(_ + _).map(_ / setNb)

    // pdf
    val distFunc = 0.5

    def hypo(x: Double) = {
      // b_bar
      a_bar * x
      // line_bar(0) + line_bar(1) * x
      // quad_bar * x * x
      // qwi_bar(0) + qwi_bar(1) * x * x
    }

    // bias
    def bias = {
      x: Double => scala.math.pow(hypo(x) - func(x), 2) * distFunc
    }
    println("bias = " + integration(bias, -1, 1))

    // TODO
//    def g(a: Double, b: Double) = {
//      x: Int =>
//        b
//        a * x
//        a * x + b
//        a * x * x
//        a * x * x + b
//    }

    // variance (just for hypo 2)
    //    def variance = {
    //      x: Double => as.map(a => scala.math.pow(a * x - hypo(x), 2)).sum / as.size * distFunc
    //    }
    //    println("variance = " + integration(variance, -1, 1))


    if (true) {
      val fig = Figure()
      val p = fig.subplot(0)

      // two extremities of domain
      val stepNb = 200
      val x = Array.tabulate(stepNb + 1)(n => -1 + 2 * n / stepNb.toDouble)

      // figure settings
      p.xlim = (-1, 1)
      p.ylim = (-2, 2)
      p.xlabel = "x"
      p.ylabel = "y"

      // draw
      p += plot(x, x.map(func), '-', "red")
      p += plot(x, x.map(hypo), '-', "blue")
    }
  }
}
