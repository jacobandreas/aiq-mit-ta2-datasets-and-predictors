"""
HELM run specifications for the darpa3 world_modeling dataset.

Run specs:
  world_modeling                              (all domains combined)
  world_modeling:domain=agent-properties
  world_modeling:domain=material-dynamics
  world_modeling:domain=material-properties
  world_modeling:domain=physical-dynamics
  world_modeling:domain=physical-interactions
  world_modeling:domain=physical-relations
  world_modeling:domain=quantitative-properties
  world_modeling:domain=social-interactions
  world_modeling:domain=social-properties
  world_modeling:domain=social-relations
  world_modeling:domain=spatial-relations
"""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec, ADAPT_GENERATION
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


_SCENARIO_CLASS = "datasets.world_modeling.scenario.WorldModelingScenario"


@run_spec_function("world_modeling")
def get_world_modeling_run_spec(domain: str = "") -> RunSpec:
    """
    Run spec for world_modeling.

    Args:
        domain: An EWOK-CORE domain name (e.g. "agent-properties"), or
                empty string / omitted for all domains combined.
    """
    effective_domain = domain if domain else None
    scenario_spec = ScenarioSpec(
        class_name=_SCENARIO_CLASS,
        args={"domain": effective_domain},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="",
        input_suffix="\n",
        output_prefix="Answer:",
        output_suffix="\n",
        max_train_instances=5,
        num_outputs=1,
        temperature=0.0,
        max_tokens=4,
        stop_sequences=["\n"],
    )

    name = "world_modeling" if not domain else f"world_modeling:domain={domain}"
    groups = ["world_modeling"]
    if domain:
        groups.append(f"world_modeling_{domain}")

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=groups,
    )
