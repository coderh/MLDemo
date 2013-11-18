package validation

import scala.math.abs
import scala.io.Source
import breeze.linalg._

/**
 *
 * Author: coderh
 * Date: 11/17/13
 * Time: 12:32 PM
 *
 */
object validation {

  case class DataPoint(x: DenseVector[Double], y: Double) {
    def this(triple: Array[Double]) {
      this(DenseVector(triple.init), triple.last)
    }

    def errorMeasure(weight: DenseVector[Double]) = {
      if (this.x.dot(weight) * this.y > 0) 0 else 1
    }
  }

  //  def extendPoint(pt: DataPoint) = {
  //    val x = DenseVector(1.0 +: pt.x.toArray)
  //    DataPoint(x, pt.y)
  //  }

  def nonLinearTrans(point: DataPoint, k: Int) = {
    require(point.x.length == 2)
    val x1 = point.x(0)
    val x2 = point.x(1)
    val fis = List(1.0, x1, x2, x1 * x1, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2))
    DataPoint(DenseVector(fis.take(k + 1).toArray), point.y)
  }


  def loadData(path: String) = {
    val data = for (line <- Source.fromFile(path).getLines()) yield
      new DataPoint(line.replaceAll("^\\s+", "").split("\\s+").map(_.toDouble))
    data.toList
  }

  def run(k: Int) = {
    val trainingSet = loadData("in.dta.txt").take(25).map(nonLinearTrans(_, k))
    val validationSet = loadData("in.dta.txt").drop(25).map(nonLinearTrans(_, k))
    val testSet = loadData("out.dta.txt").map(nonLinearTrans(_, k))

    val D = k + 1

    val X = DenseMatrix.tabulate(trainingSet.size, D)((i: Int, j: Int) => trainingSet(i).x(j))
    val Xt = X.t
    val y = DenseVector(trainingSet.map(_.y).toArray)
    val w = inv(Xt * X) * Xt * y

    val E_val = validationSet.map(_.errorMeasure(w)).count(_ == 1).toDouble / validationSet.size.toDouble
    val E_out = testSet.map(_.errorMeasure(w)).count(_ == 1).toDouble / testSet.size.toDouble

    (E_val, E_out)
  }

  def main(args: Array[String]) = {
    val res = 3 to 7 map run map (_._2)
    println(res)
  }

}
