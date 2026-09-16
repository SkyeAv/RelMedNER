package relmedner

import org.apache.beam.sdk.expansion.service.ExpansionService

object ScalaExpander {
  val expansionServicePort = Array("9097")

  @main
  def launchBeamRunner() = {
    ExpansionService.main(expansionServicePort)
  }
}
