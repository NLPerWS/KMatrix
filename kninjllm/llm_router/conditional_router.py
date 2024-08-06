import json
from typing import Any, Dict, List, Set

from jinja2 import Environment, TemplateSyntaxError, meta
from jinja2.nativetypes import NativeEnvironment

from kninjllm.llm_common.component import component
from kninjllm.llm_common.serialization import default_to_dict
from kninjllm.llm_utils.type_serialization_utils import deserialize_type, serialize_type

class NoRouteSelectedException(Exception):
    """Exception raised when no route is selected in ConditionalRouter."""


class RouteConditionException(Exception):
    """Exception raised when there is an error parsing or evaluating the condition expression in ConditionalRouter."""


@component
class ConditionalRouter:

    def __init__(self,max_loop_count, routes: List[Dict]):
        """
        Initializes the `ConditionalRouter` with a list of routes detailing the conditions for routing.

        :param routes: A list of dictionaries, each defining a route.
            A route dictionary comprises four key elements:
            - `condition`: A Jinja2 string expression that determines if the route is selected.
            - `output`: A Jinja2 expression defining the route's output value.
            - `output_type`: The type of the output data (e.g., str, List[int]).
            - `output_name`: The name under which the `output` value of the route is published. This name is used to connect
            the router to other components in the pipeline.
        """
        self._validate_routes(routes)
        
        self.routes: List[dict] = routes
        self.max_loop_count = max_loop_count
        self.temp_loop_count = 1
        # Create a Jinja native environment to inspect variables in the condition templates
        env = NativeEnvironment()

        # Inspect the routes to determine input and output types.
        input_types: Set[str] = set()  # let's just store the name, type will always be Any
        output_types: Dict[str, str] = {}

        for route in routes:
            # extract inputs
            route_input_names = self._extract_variables(env, [route["output"], route["condition"]])
            input_types.update(route_input_names)

            # extract outputs
            output_types.update({route["output_name"]: route["output_type"]})

        component.set_input_types(self, **{var: Any for var in input_types})
        component.set_output_types(self, **output_types)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        for route in self.routes:
            # output_type needs to be serialized to a string
            route["output_type"] = serialize_type(route["output_type"])

        return default_to_dict(self, routes=self.routes)


    def run(self,max_loop_count = 0,temp_loop_count = 0,**kwargs):
        env = NativeEnvironment()
        kwargs['max_loop_count'] = self.max_loop_count
        kwargs['temp_loop_count'] = self.temp_loop_count

        for route in self.routes:
            try:
                t = env.from_string(route["condition"])
                if t.render(**kwargs):
                    t_output = env.from_string(route["output"])
                    output = t_output.render(**kwargs)
                    self.temp_loop_count += 1
                    return {route["output_name"]: output}
            except Exception as e:
                raise RouteConditionException(f"Error evaluating condition for route '{route}': {e}") from e

        raise NoRouteSelectedException(f"No route fired. Routes: {self.routes}")

    def _validate_routes(self, routes: List[Dict]):
        """
        Validates a list of routes.

        :param routes: A list of routes.
        """
        env = NativeEnvironment()
        for route in routes:
            try:
                keys = set(route.keys())
            except AttributeError:
                raise ValueError(f"Route must be a dictionary, got: {route}")

            mandatory_fields = {"condition", "output", "output_type", "output_name"}
            has_all_mandatory_fields = mandatory_fields.issubset(keys)
            if not has_all_mandatory_fields:
                raise ValueError(
                    f"Route must contain 'condition', 'output', 'output_type' and 'output_name' fields: {route}"
                )
            for field in ["condition", "output"]:
                if not self._validate_template(env, route[field]):
                    raise ValueError(f"Invalid template for field '{field}': {route[field]}")

    def _extract_variables(self, env: NativeEnvironment, templates: List[str]) -> Set[str]:
        """
        Extracts all variables from a list of Jinja template strings.

        :param env: A Jinja environment.
        :param templates: A list of Jinja template strings.
        :returns: A set of variable names.
        """
        variables = set()
        for template in templates:
            ast = env.parse(template)
            variables.update(meta.find_undeclared_variables(ast))
        return variables

    def _validate_template(self, env: Environment, template_text: str):
        """
        Validates a template string by parsing it with Jinja.

        :param env: A Jinja environment.
        :param template_text: A Jinja template string.
        :returns: `True` if the template is valid, `False` otherwise.
        """
        try:
            env.parse(template_text)
            return True
        except TemplateSyntaxError:
            return False