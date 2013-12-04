package radial_basis_function

import breeze.linalg.{DenseMatrix, DenseVector, inv}
import scala.util.Random
import scala.math._
import libsvm._
import libsvm.svm._
import scala.Array
import breeze.plot._

/**
 *
 * Author: coderh
 * Date: 12/3/13
 * Time: 9:40 PM
 *
 */


object RBF {
  type DataSet = List[DataPoint]


  def targetFunc(x: DenseVector[Double]): Double = {
    require(x.length == 2)
    val x1 = x(0)
    val x2 = x(1)
    if (x2 - x1 + 0.25 * sin(Pi * x1) > 0) 1 else -1
  }

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

  def generateDataPoint(dim: Int) = {
    val x = DenseVector.tabulate(dim)(_ => 2 * Random.nextDouble() - 1)
    DataPoint(x, targetFunc(x))
  }

  def vector2svmNode(v: DenseVector[Double]) = Array(new svm_node() {
    this.index = 1
    this.value = v(0)
  }, new svm_node() {
    this.index = 2
    this.value = v(1)
  })

  def Lloyd(data: DataSet, K: Int): DataSet = {

    val dim = 2
    val labeledDataSet = data.map(pt => DataPoint(pt.x, -1))

    // init centers
    var centers = List.tabulate(K)(i => DataPoint(DenseVector.tabulate(dim)(_ => 2 * Random.nextDouble() - 1), i))

    // init clusters
    var clusteredDataSet: DataSet = Nil

    // var temp
    var centers_old: DataSet = Nil

    def isCenterChanged(centerSet1: DataSet, centerSet2: DataSet) = {
      centerSet1.sortBy(_.y) zip centerSet2.sortBy(_.y) exists (p => (p._1.x - p._2.x).sum != 0)
    }

    var discard = false

    do {

      centers_old = centers
      clusteredDataSet = labeledDataSet map {
        case pt =>
          val distList = centers.map(center => ((center.x - pt.x).dot(center.x - pt.x), center.y))
          val clusterLable = distList.minBy(_._1)._2
          DataPoint(pt.x, clusterLable)
      }

      // test all cluster exist
      if (clusteredDataSet.groupBy(_.y).size != K) {
        discard = true
      } else {
        centers = (clusteredDataSet groupBy (_.y) map {
          case pair =>
            val mean = pair._2.map(_.x).reduce[DenseVector[Double]](_ + _) / pair._2.size.toDouble
            DataPoint(mean, pair._1)
        }).toList
      }


    } while (isCenterChanged(centers_old, centers) && !discard)

    if (discard) Nil else centers.sortBy(_.y).toList
  }

  def RBF_regular(trainingSet: DataSet, testSet: DataSet, K: Int, gamma: Double) = {
    val N = trainingSet.size
    var cter = Lloyd(trainingSet, K)

    while (cter.isEmpty) {
      cter = Lloyd(trainingSet, K)
    }

    //    plotData(trainingSet, centers = cter)

    val X = DenseMatrix.tabulate(N, K + 1)((i: Int, j: Int) => {
      if (j == 0) 1 else exp(-gamma * (trainingSet(i).x - cter(j - 1).x).dot(trainingSet(i).x - cter(j - 1).x))
    })
    val Xt = X.t
    val Y = DenseVector(trainingSet.map(_.y).toArray)
    val w = inv(Xt * X) * Xt * Y

    def hypo(x: DenseVector[Double]) = {
      val basisFuncList = DenseVector(1.0 +: cter.map(ct => exp(-gamma * (x - ct.x).dot(x - ct.x))).toArray)
      if (w.dot(basisFuncList) > 0) 1 else -1
    }

    val E_in = trainingSet.count(pt => hypo(pt.x) * pt.y < 0) / trainingSet.size.toDouble
    val E_out = testSet.count(pt => hypo(pt.x) * pt.y < 0) / testSet.size.toDouble
    (E_in, E_out)
  }

  def plotData(dataSet: List[DataPoint],
               centers: List[DataPoint] = Nil,
               SVs: List[DenseVector[Double]] = Nil) = {

    val f = Figure()
    val p = f.subplot(0)

    // figure settings
    p.xlim = (-1, 1)
    p.ylim = (-1, 1)
    p.xlabel = "x"
    p.ylabel = "y"

    // two extremities of domain
    //    val xs = Array(-1.0, 1.0)

    val xs: Array[Double] = Array.tabulate(1000)(x => -1.0 + 2.0 / 1000 * x)

    // draw f(x)
    p += plot(xs, xs.map(x => x - 0.25 * sin(Pi * x)), '-', "r")

    // draw data
    val oneSide = dataSet.filter(_.y > 0).map(_.x)
    val theOtherSide = dataSet.filter(_.y < 0).map(_.x)
    p += plot(oneSide.map(_(0)), oneSide.map(_(1)), '.', "g")
    p += plot(theOtherSide.map(_(0)), theOtherSide.map(_(1)), '.', "b")

    if (!centers.isEmpty) {
      p += plot(centers.map(_.x(0)), centers.map(_.x(1)), '+', "m")
    }

    if (!SVs.isEmpty) {
      p += plot(SVs.map(_(0)), SVs.map(_(1)), '+', "y")
    }
  }

  def kernel(SV: DenseVector[Double])(X: DenseVector[Double]) = {
    val gamma = 1.5
    exp(-gamma * (SV - X).dot(SV - X))
  }

  def generateDataSet(N: Int) = {
    val D = 2
    List.tabulate(N)(_ => generateDataPoint(D))
  }

  def RBF_kernel(trainingSet: DataSet, testSet: DataSet) = {

    val prob = new svm_problem() {
      this.l = trainingSet.size
      this.y = trainingSet.map(_.y).toArray
      this.x = trainingSet.map(_.x).map {
        case v: DenseVector[Double] =>
          Array(new svm_node() {
            this.index = 1
            this.value = v(0)
          }, new svm_node() {
            this.index = 2
            this.value = v(1)
          })
      }.toArray
    }

    val param = new svm_parameter {
      this.svm_type = svm_parameter.C_SVC
      this.kernel_type = svm_parameter.RBF
      this.gamma = 1.5
      this.eps = 0.001
      this.nr_weight = 0
      this.cache_size = 10
      this.C = Double.PositiveInfinity
    }

    //disable display
    val display = new svm_print_interface() {
      def print(s: String) {
      }
    }
    svm.svm_set_print_string_function(display)

    val check = svm.svm_check_parameter(prob, param)
    val model = if (check == null) svm.svm_train(prob, param) else throw new Error("Wrong SVM parameters: " + check)

    val SVs = model.SV.toList.map {
      case xs: Array[svm_node] => DenseVector[Double](xs(0).value, xs(1).value)
    }

    //    plotData(trainingSet, SVs)

    //    val E_in = trainingSet.count(pt => svm_predict(model, vector2svmNode(pt.x)) * pt.y < 0).toDouble / trainingSet.size
    val E_out = testSet.count(pt => svm_predict(model, vector2svmNode(pt.x)) * pt.y < 0).toDouble / testSet.size

    //    val b = model.rho.head
    //    val coef = model.sv_coef.head.toList
    //    val refPoint = trainingSet.find(_.x == SVs.head) match {
    //      case Some(x) => x
    //      case None => throw new Error("SV is not in training set")
    //    }
    //    val b = refPoint.y - (coef zip SVs).map(p => p._1 * kernel(p._2)(refPoint.x)).sum
    //
    //    def g(x: DenseVector[Double]) = {
    //      val r = (coef zip SVs).map(p => p._1 * kernel(p._2)(x)).sum + b
    //      if (r > 0) 1 else -1
    //    }
    //
    //    val E_in = trainingSet.map(pt => pt.y * g(pt.x)).count(_ < 0).toDouble / trainingSet.size
    //    (b, bb)
    E_out
  }

  def solution() = {
    val trainingSet = generateDataSet(100)
    val testSet = generateDataSet(10000)
    val K = 9
    val gamma = 1.5
    //    val kernelPerf = RBF_kernel(trainingSet, testSet)
    val (e_in, e_out) = RBF_regular(trainingSet, testSet, K, gamma)

    // cases:
    //    val a = if (perf2._1 < perf1._1 && perf2._2 > perf1._2) 1 else 0
    //    val b = if (perf2._1 > perf1._1 && perf2._2 < perf1._2) 1 else 0
    //    val c = if (perf2._1 > perf1._1 && perf2._2 > perf1._2) 1 else 0
    //    val d = if (perf2._1 < perf1._1 && perf2._2 < perf1._2) 1 else 0
    //    val e = if (perf2._1 == perf1._1 && perf2._2 == perf1._2) 1 else 0
    //    DenseVector(a, b, c, d, e)
    e_in
  }


  def main(args: Array[String]) = {
    //    solution
    val iter = 1000
    //    val perc = (1 to iter).map(_ => solution).count(_ < 0) / iter.toDouble
    val v = (1 to iter).map(_ => solution).count(_ == 0) / iter.toDouble
    println(v)
  }
}
