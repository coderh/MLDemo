package overfitting_regularization_with_weight_decay

import scala.io.Source
import scala.math.{abs, pow}
import breeze.linalg._

/**
 *
 * Author: coderh
 * Date: 11/11/13
 * Time: 10:53 AM
 *
 */
object regularization {

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

  def nonLinearTrans(point: DataPoint) = {
    require(point.x.length == 2)
    val x1 = point.x(0)
    val x2 = point.x(1)
    DataPoint(DenseVector(1, x1, x2, x1 * x1, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)), point.y)
  }


  def loadData(path: String) = {
    val data = for (line <- Source.fromFile(path).getLines()) yield
      new DataPoint(line.replaceAll("^\\s+", "").split("\\s+").map(_.toDouble))
    data.toList
  }


  def run() {
    val trainingSet = loadData("in.dta.txt").map(nonLinearTrans)
    val testSet = loadData("out.dta.txt").map(nonLinearTrans)

    // nb of features
    val D = 8

    val X = DenseMatrix.tabulate(trainingSet.size, D)((i: Int, j: Int) => trainingSet(i).x(j))
    val Xt = X.t
    val y = DenseVector(trainingSet.map(_.y).toArray)
    val w = inv(Xt * X) * Xt * y

    println("=== Without regularization(weight decay) ===")

    // in-sample error
    val E_in = trainingSet.map(_.errorMeasure(w)).count(_ == 1).toDouble / trainingSet.size.toDouble
    println("in-sample error is: " + E_in)

    // out-of-sample error
    val E_out = testSet.map(_.errorMeasure(w)).count(_ == 1).toDouble / testSet.size.toDouble
    println("out-of-sample error is: " + E_out)

    println("\n=== With regularization(weight decay) ===")

    val k = -2
    val lambda = pow(10, k)
    val eye = DenseMatrix.eye[Double](D)
    val w_withRegul = inv(Xt * X + eye * lambda) * Xt * y

    // in-sample error
    val E_in_withRegul = trainingSet.map(_.errorMeasure(w_withRegul)).count(_ == 1).toDouble / trainingSet.size.toDouble
    println("in-sample error is: " + E_in_withRegul)

    // out-of-sample error
    val E_out_withRegul = testSet.map(_.errorMeasure(w_withRegul)).count(_ == 1).toDouble / testSet.size.toDouble
    println("out-of-sample error is: " + E_out_withRegul)
  }

  def main(args: Array[String]) {
    //run()
    val x = 22
    val res = -x * x + 44 * x + 26
    println(res)
  }
}
