"""
HELM run specifications for the darpa3 arithmetic_fixed dataset.

Run specs:
  arithmetic_fixed:split_config=cross_lingual
"""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec, ADAPT_GENERATION
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


_SCENARIO_CLASS = "datasets.arithmetic_fixed.scenario.ArithmeticFixedScenario"


@run_spec_function("arithmetic_fixed")
def get_arithmetic_fixed_run_spec(split_config: str = "cross_lingual") -> RunSpec:
    """
    Run spec for arithmetic_fixed.

    Args:
        split_config: Name of the split configuration under data/arithmetic_fixed/.
                      Currently only "cross_lingual" is defined (train: english+symbolic,
                      test: spanish+italian).
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
        max_train_instances=1,
        num_outputs=1,
        temperature=0.0,
        max_tokens=32,
        stop_sequences=["\n"],
    )

    return RunSpec(
        name=f"arithmetic_fixed:split_config={split_config}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["arithmetic_fixed", f"arithmetic_fixed_{split_config}"],
    )
