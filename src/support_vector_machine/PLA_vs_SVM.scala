package support_vector_machine

import scala.util.Random
import breeze.linalg.{DenseMatrix, DenseVector, inv}
import breeze.plot._

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD


/**
 *
 * Author: coderh
 * Date: 11/17/13
 * Time: 6:08 PM
 *
 */

trait experiment {


  case class DataPoint(x: DenseVector[Double], y: Double) {
    def this(arr: Array[Double]) {
      this(DenseVector(arr.init), arr.last)
    }

    def errorMeasure(weight: DenseVector[Double]) = {
      if (this.x.dot(weight) * this.y > 0) 0 else 1
    }
  }

  def extendPoint(pt: DataPoint) = {
    val x = DenseVector(1.0 +: pt.x.toArray)
    DataPoint(x, pt.y)
  }

  def generateDataPoint(dim: Int, targetFunc: DenseVector[Double] => Double) = {
    val x = DenseVector.tabulate(dim)(_ => 2 * Random.nextDouble() - 1)
    DataPoint(x, targetFunc(DenseVector(1.0 +: x.toArray)))
  }

}

object PLA_vs_SVM extends experiment {

  // TODO: correct it!
  // settings
  val pt1 = DenseVector(2 * Random.nextDouble() - 1, 2 * Random.nextDouble() - 1)
  val pt2 = DenseVector(2 * Random.nextDouble() - 1, 2 * Random.nextDouble() - 1)
  val w2 = pt2(0) - pt1(0)
  val w1 = pt1(1) - pt2(1)
  val w0 = pt2(1) * pt1(0) - pt1(1) * pt2(0)
  val tf = (x: DenseVector[Double]) => if ((DenseVector(w0, w1, w2) dot x) > 0) 1.0 else -1.0

  // dimension
  val d = 2

  // nb of data points
  val N = 10

  // generate data sets
  var dataSet = List.tabulate(N)(_ => generateDataPoint(d, tf))
  while (dataSet.forall(_.y > 0) || dataSet.forall(_.y < 0)) {
    dataSet = List.tabulate(N)(_ => generateDataPoint(d, tf))
  }
  val extendedDataSet = dataSet.map(extendPoint)



//  val sc = new SparkContext("local[2]", "SparkLR", System.getenv("SPARK_HOME"), null)

  object SVM {

//    def run_spark() = {
//
//
//      val parsedData = dataSet.map(pt => LabeledPoint(if (pt.y > 0) 1 else 0, pt.x.toArray)).toSeq
//      val rddData = sc.parallelize(parsedData)
//
//      val numIterations = 30
//      val model = SVMWithSGD.train(rddData, numIterations)
//
//      val n = 100000
//      val testData = sc.parallelize(Array.tabulate(n)(_ => generateDataPoint(d, tf)).map(pt => LabeledPoint(if (pt.y > 0) 1 else 0, pt.x.toArray)).toSeq, 2)
//
//      val labelAndPreds = testData.map {
//        point =>
//          val prediction = model.predict(point.features)
//          (point.label, prediction)
//      }
//
//      val err = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / n
//      err
//    }

    def run() = {

      val C = DenseVector.tabulate(N)(_ => -1.0)
      val Q = DenseMatrix.tabulate(N, N)((i: Int, j: Int) => (dataSet(i).x dot dataSet(j).x) * dataSet(i).y * dataSet(j).y)

      //      var Z = DenseMatrix.tabulate(N, N)((i: Int, j: Int) => Random.nextDouble())
      val Z = Q
      //    val Z = DenseMatrix.eye[Double](N)
      val alpha_prim = inv(Z.t * Q * Z) * (-Z.t * C)
      val alpha = Z * alpha_prim

      //      while (!alpha.toArray.forall(_ >= 0)) {
      //        Z = DenseMatrix.tabulate(N, N)((i: Int, j: Int) => Random.nextDouble())
      //        //    val Z = DenseMatrix.eye[Double](N)
      //        alpha_prim = -inv(Z.t * Q * Z) * (-Z.t * C)
      //        alpha = Z * alpha_prim
      //      }
      alpha
    }
  }

  object PLA {

    def run() = {

      var w = DenseVector[Double](0, 0, 0)

      def isMisclassified(pt: DataPoint) = (pt.x dot w) * pt.y <= 0

      var misclassifiedSet = extendedDataSet.filter(isMisclassified)
      while (!misclassifiedSet.isEmpty) {
        val missedPt = Random.shuffle(misclassifiedSet.toList).head
        w = w + missedPt.x * missedPt.y
        misclassifiedSet = extendedDataSet.filter(isMisclassified)
      }

      if (false) {

        val f = Figure()
        val p = f.subplot(0)

        // figure settings
        p.xlim = (-1, 1)
        p.ylim = (-1, 1)
        p.xlabel = "x"
        p.ylabel = "y"

        // two extremities of domain
        val xs = Array(-1.0, 1.0)

        // draw f(x)
        p += plot(xs, xs.map(x => (w0 + w1 * x) / -w2), '-', "red")

        // draw g(x)
        p += plot(xs, xs.map(x => (w(0) + w(1) * x) / -w(2)), '-', "blue")

        // draw data
        val oneSide = dataSet.filter(_.y > 0).map(_.x)
        val theOtherSide = dataSet.filter(_.y < 0).map(_.x)
        p += plot(oneSide.map(_(0)), oneSide.map(_(1)), '+')
        p += plot(theOtherSide.map(_(0)), theOtherSide.map(_(1)), '.')
      }

      // error
      val n = 100000
      //    val err = sc.parallelize(Array.tabulate(n)(_ => generatePoint), 2).cache().filter(isMisclassified).count.toDouble / n
      val err = List.tabulate(n)(_ => generateDataPoint(d, tf)).map(extendPoint).count(isMisclassified).toDouble / n

      err
      //    println(w)
      //    println(err)
    }


  }


  def main(args: Array[String]) = {
    var a = SVM.run()
    while(!a.toArray.forall(_ >= 0)){
      a = SVM.run()
    }

    println(a)
    //    val iter = 1000
    //    val perf_pla = for (i <- 1 to iter) yield PLA.run()
    //    val perf_svm = for (i <- 1 to iter) yield SVM.run_spark()
    //    val percentage = (perf_pla zip perf_svm).count(p => p._1 > p._2).toDouble / iter
    //    println(percentage)
  }
}