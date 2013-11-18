package support_vector_machine

import scala.util.Random
import breeze.linalg.DenseVector
import breeze.plot._


/**
 *
 * Author: coderh
 * Date: 11/17/13
 * Time: 6:08 PM
 *
 */
object PLA_vs_SVM {

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

  object SVM extends experiment {

    def run(N: Int) = {
      0
    }
  }

  object PLA extends experiment {

    def run(N: Int) = {

      val pt1 = DenseVector(2 * Random.nextDouble() - 1, 2 * Random.nextDouble() - 1)
      val pt2 = DenseVector(2 * Random.nextDouble() - 1, 2 * Random.nextDouble() - 1)
      val w2 = pt2(0) - pt1(0)
      val w1 = pt1(1) - pt2(1)
      val w0 = pt2(1) * pt1(0) - pt1(1) * pt2(0)
      val tf = (x: DenseVector[Double]) => if ((DenseVector(w0, w1, w2) dot x) > 0) 1.0 else -1.0

      val d = 2


      var dataSet = List.tabulate(N)(_ => generateDataPoint(d, tf))
      while (dataSet.forall(_.y > 0) || dataSet.forall(_.y < 0)) {
        dataSet = List.tabulate(N)(_ => generateDataPoint(d, tf))
      }
      val extendedDataSet = dataSet.map(extendPoint)

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
    val N = 10
    val iter = 1
    val perf_pla = for (i <- 1 to iter) yield PLA.run(N)
    val perf_svm = for (i <- 1 to iter) yield SVM.run(N)
    val percentage = (perf_pla zip perf_svm).count(p => p._1 > p._2).toDouble / iter
    println(percentage)
  }
}