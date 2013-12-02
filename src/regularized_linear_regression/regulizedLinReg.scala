package regularized_linear_regression

import breeze.linalg._
import scala.io.Source
import scala.Array

/**
 *
 * Author: coderh
 * Date: 12/2/13
 * Time: 10:31 PM
 *
 */
object regulizedLinReg {

  case class DataPoint(x: DenseVector[Double], y: Double) {
    def this(arr: Array[Double]) {
      this(DenseVector(arr.tail), arr.head)
    }

    def errorMeasure(weight: DenseVector[Double]) = {
      if (this.x.dot(weight) * this.y > 0) 0 else 1
    }
  }

  def extendPoint(pt: DataPoint) = {
    val x = DenseVector(1.0 +: pt.x.toArray)
    DataPoint(x, pt.y)
  }

  def loadData(path: String) = {
    val data = for (line <- Source.fromFile(path).getLines()) yield
      new DataPoint(line.replaceAll("^\\s+", "").split("\\s+").map(_.toDouble))
    data.toList
  }

  def XvsAll(pt: DataPoint, x: Double) = {
    val label = if (pt.y == x) 1 else -1
    DataPoint(pt.x, label)
  }

  def nonLinearTrans(point: DataPoint) = {
    require(point.x.length == 2)
    val x1 = point.x(0)
    val x2 = point.x(1)
    DataPoint(DenseVector(1, x1, x2, x1 * x2, x1 * x1, x2 * x2), point.y)
  }

  def run(label: Int) = {
    //  def run(lambda: Double) = {

    val lambda = 1d

    val trainingSet = loadData("features.train.txt").map(pt => XvsAll(pt, label))
    val testSet = loadData("features.test.txt").map(pt => XvsAll(pt, label))

    val extendedTrainingSet = trainingSet map extendPoint
    val extendedTestSet = testSet map extendPoint

    //    val trainingSet = loadData("features.train.txt").filter(pt => pt.y == 1.00 || pt.y == 5.00).map(pt => XvsAll(pt, 1))
    //    val testSet = loadData("features.test.txt").filter(pt => pt.y == 1.00 || pt.y == 5.00).map(pt => XvsAll(pt, 1))

    val transTrainingSet = trainingSet map nonLinearTrans
    val transTestSet = testSet map nonLinearTrans

    val (err_in_ext, err_out_ext) = {

      val D = 3

      val Z = DenseMatrix.tabulate(extendedTrainingSet.size, D)((i: Int, j: Int) => extendedTrainingSet(i).x(j))
      val Zt = Z.t
      val y = DenseVector(extendedTrainingSet.map(_.y).toArray)

      val eye = DenseMatrix.eye[Double](D)
      val w_reg = inv(Zt * Z + eye * lambda) * Zt * y

      // in-sample error
      val E_in = extendedTrainingSet.map(_.errorMeasure(w_reg)).count(_ == 1).toDouble / extendedTrainingSet.size.toDouble

      // out-of-sample error
      val E_out = extendedTestSet.map(_.errorMeasure(w_reg)).count(_ == 1).toDouble / extendedTestSet.size.toDouble
      (E_in, E_out)
    }

    val (err_in_trans, err_out_trans) = {

      val D = 6

      val Z = DenseMatrix.tabulate(transTrainingSet.size, D)((i: Int, j: Int) => transTrainingSet(i).x(j))
      val Zt = Z.t
      val y = DenseVector(transTrainingSet.map(_.y).toArray)

      val eye = DenseMatrix.eye[Double](D)
      val w_reg = inv(Zt * Z + eye * lambda) * Zt * y

      // in-sample error
      val E_in = transTrainingSet.map(_.errorMeasure(w_reg)).count(_ == 1).toDouble / transTrainingSet.size.toDouble

      // out-of-sample error
      val E_out = transTestSet.map(_.errorMeasure(w_reg)).count(_ == 1).toDouble / transTestSet.size.toDouble
      (E_in, E_out)
    }

    println(label + " : " + List(err_in_ext, err_in_trans, err_out_ext, err_out_trans).map(x => "%.5f".format(x)).mkString("\t\t"))
    //    println(label + " : " + List(err_in_ext, err_in_trans, err_out_ext, err_out_trans).mkString("\t\t"))
    //    println(label + " : " + (err_out_trans - err_out_ext) / err_out_ext)
    //    println(lambda + " : " + List(err_in_trans, err_out_trans).mkString("\t\t"))
    println
  }


  def main(args: Array[String]) = {
    0 to 9 foreach run
    //    List(0.01, 1d) foreach run
  }

}
