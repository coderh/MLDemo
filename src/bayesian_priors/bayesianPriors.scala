package bayesian_priors

import scala.util.Random

/**
 * Created with IntelliJ IDEA.
 * User: coderh
 * Date: 12/4/13
 * Time: 2:22 PM
 */
object bayesianPriors {
  val hSet = List(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

  def experiment(h: Double) = {

    def sample() = {
      val N = 1000
      Random.shuffle((1 to N).map(_ => Random.nextDouble() <= h)).head
    }

    val iter = 100000
    (1 to iter).map(_ => sample).count(_ == true) / iter.toDouble
  }

  def main(args: Array[String]) = {
    println(hSet map experiment)
  }
}
