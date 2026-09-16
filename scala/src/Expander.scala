package relmedner
import scalatags.Text.all.*

object Expander {
  def generateHtml(text: String) = {
    h1(text).toString
  }

  @main
  def main() = {
    println(generateHtml("hi"))
  }
}
