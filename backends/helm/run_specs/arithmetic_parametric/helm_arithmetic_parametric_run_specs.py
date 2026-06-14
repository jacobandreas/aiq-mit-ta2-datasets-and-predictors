"""
HELM run specifications for the darpa3 arithmetic_parametric dataset.

Run specs:
  arithmetic_parametric:split_config=arithmetic_by_format
  arithmetic_parametric:split_config=arithmetic_by_format,eval_split=val
  arithmetic_parametric:split_config=arithmetic_by_skill
  arithmetic_parametric:split_config=arithmetic_by_skill,eval_split=val
"""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec, ADAPT_GENERATION
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


_SCENARIO_CLASS = "datasets.arithmetic_parametric.scenario.ArithmeticParametricScenario"


@run_spec_function("arithmetic_parametric")
def get_arithmetic_parametric_run_spec(split_config: str = "arithmetic_by_format") -> RunSpec:
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
        name=f"arithmetic_parametric:split_config={split_config}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["arithmetic_parametric", split_config],
    )
