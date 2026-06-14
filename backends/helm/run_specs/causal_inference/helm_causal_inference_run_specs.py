"""
HELM run specifications for the darpa3 causal_inference dataset.

Run spec:
  causal_inference:split_config=causal_default
"""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec, ADAPT_GENERATION
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


_SCENARIO_CLASS = "datasets.causal_inference.scenario.CausalInferenceScenario"


@run_spec_function("causal_inference")
def get_causal_inference_run_spec(
    split_config: str = "causal_default",
) -> RunSpec:
    """
    Run spec for causal_inference.

    Args:
        split_config: Currently only "causal_default".
    """
    scenario_spec = ScenarioSpec(
        class_name=_SCENARIO_CLASS,
        args={"split_config": split_config},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="",
        input_suffix="",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=5,
        num_outputs=1,
        temperature=0.0,
        max_tokens=16,
        stop_sequences=["\n"],
    )

    return RunSpec(
        name=f"causal_inference:split_config={split_config}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["causal_inference", split_config],
    )
