import datetime
import logging
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import posthog
import yaml

from kninjllm.llm_common.serialization import generate_qualified_class_name
from kninjllm.llm_telemetry._environment import collect_system_specs

if TYPE_CHECKING:
    from kninjllm.llm_pipeline.pipeline import Pipeline


KNINJLLM_TELEMETRY_ENABLED = "KNINJLLM_TELEMETRY_ENABLED"
CONFIG_PATH = Path("~/.kninjllm/config.yaml").expanduser()

#: Telemetry sends at most one event every number of seconds specified in this constant
MIN_SECONDS_BETWEEN_EVENTS = 60



class Telemetry:

    def __init__(self):
        """
        Initializes the telemetry. Loads the user_id from the config file,
        or creates a new id and saves it if the file is not found.

        It also collects system information which cannot change across the lifecycle
        of the process (for example `is_containerized()`).
        """
        posthog.api_key = "phc_C44vUK9R1J6HYVdfJarTEPqVAoRPJzMXzFcj8PIrJgP"
        posthog.host = "https://eu.posthog.com"

        self.user_id = None

        if CONFIG_PATH.exists():
            # Load the config file
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
                    config = yaml.safe_load(config_file)
                    if "user_id" in config:
                        self.user_id = config["user_id"]
            except Exception as e:
                psdd
        else:
            # Create the config file
            CONFIG_PATH.parents[0].mkdir(parents=True, exist_ok=True)
            self.user_id = str(uuid.uuid4())
            try:
                with open(CONFIG_PATH, "w") as outfile:
                    yaml.dump({"user_id": self.user_id}, outfile, default_flow_style=False)
            except Exception as e:
                pass

        self.event_properties = collect_system_specs()

    def send_event(self, event_name: str, event_properties: Optional[Dict[str, Any]] = None):
        """
        Sends a telemetry event.

        :param event_name: The name of the event to show in PostHog.
        :param event_properties: Additional event metadata. These are merged with the
            system metadata collected in __init__, so take care not to overwrite them.
        """
        event_properties = event_properties or {}
        try:
            posthog.capture(
                distinct_id=self.user_id, event=event_name, properties={**self.event_properties, **event_properties}
            )
        except Exception as e:
            pass


def send_telemetry(func):
    """
    Decorator that sends the output of the wrapped function to PostHog.
    The wrapped function is actually called only if telemetry is enabled.
    """

    # FIXME? Somehow, functools.wraps makes `telemetry` out of scope. Let's take care of it later.
    def send_telemetry_wrapper(*args, **kwargs):
        try:
            if telemetry:
                output = func(*args, **kwargs)
                if output:
                    telemetry.send_event(*output)
        except Exception as e:
            # Never let telemetry break things
            pass

    return send_telemetry_wrapper


@send_telemetry
def pipeline_running(pipeline: "Pipeline") -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Collects name, type and the content of the _telemetry_data attribute, if present, for each component in the
    pipeline and sends such data to Posthog.

    :param pipeline: the pipeline that is running.
    """
    pipeline._telemetry_runs += 1
    if (
        pipeline._last_telemetry_sent
        and (datetime.datetime.now() - pipeline._last_telemetry_sent).seconds < MIN_SECONDS_BETWEEN_EVENTS
    ):
        return None

    pipeline._last_telemetry_sent = datetime.datetime.now()

    # Collect info about components
    components: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for component_name, instance in pipeline.walk():
        component_qualified_class_name = generate_qualified_class_name(type(instance))
        if hasattr(instance, "_get_telemetry_data"):
            telemetry_data = getattr(instance, "_get_telemetry_data")()
            if not isinstance(telemetry_data, dict):
                raise TypeError(
                    f"Telemetry data for component {component_name} must be a dictionary but is {type(telemetry_data)}."
                )
            components[component_qualified_class_name].append({"name": component_name, **telemetry_data})
        else:
            components[component_qualified_class_name].append({"name": component_name})

    # Data sent to Posthog
    return "Pipeline run (2.x)", {
        "pipeline_id": str(id(pipeline)),
        "runs": pipeline._telemetry_runs,
        "components": components,
    }


@send_telemetry
def tutorial_running(tutorial_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Send a telemetry event for a tutorial, if telemetry is enabled.
    :param tutorial_id: identifier of the tutorial
    """
    return "Tutorial", {"tutorial.id": tutorial_id}


telemetry = None
if os.getenv("KNINJLLM_TELEMETRY_ENABLED", "true").lower() in ("true", "1"):
    telemetry = Telemetry()
