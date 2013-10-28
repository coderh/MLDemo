package bias_and_variance

import scala.math._
import breeze.linalg._
import breeze.plot._
import scala.util.Random

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

  def generate2Point = {
    val x1 = 2 * Random.nextDouble - 1
    val x2 = 2 * Random.nextDouble - 1
    (DataPoint(x1, func(x1)), DataPoint(x2, func(x2)))
  }

  def getSlope(pts: (DataPoint, DataPoint)) = {
//    (pts._1.y + pts._2.y) / (pts._1.x + pts._2.y)
    (pts._1.x * pts._1.y + pts._2.x * pts._2.y) / (pts._1.x * pts._1.x + pts._2.x * pts._2.x)
  }

  def main(args: Array[String]) {

    val setNb = 100000
    val dataSets = List.tabulate(setNb)(_ => generate2Point)
    val a = dataSets.map(getSlope).sum / setNb
    //    println(dataSets)
    println(a)

    /*
    val fig = Figure()
    val p = fig.subplot(0)

    // figure settings
    p.xlim = (-1, 1)
    p.ylim = (-1.2, 1.2)
    p.xlabel = "x"
    p.ylabel = "y"

    // two extremities of domain
    val stepNb = 200
    val x = Array.tabulate(stepNb + 1)(n => -1 + 2 * n / stepNb.toDouble)

    // draw f(x)
    p += plot(x, x.map(func), '-', "red")
    p += plot(x, x.map(_ * a), '-', "blue")
     //*/
  }
}
