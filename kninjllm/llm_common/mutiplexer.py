import json
import sys
from typing import Any, Dict, List

from kninjllm.llm_common.component import component
from kninjllm.llm_common.serialization import default_from_dict, default_to_dict
from kninjllm.llm_common.types import Variadic
from kninjllm.llm_utils.type_serialization_utils import deserialize_type, serialize_type

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias


@component(is_greedy=True)
class Multiplexer:

    def __init__(self, type_: Any=Any):
        """
        Create a `Multiplexer` component.

        :param type_: The type of data that the `Multiplexer` will receive from the upstream connected components and
                        distribute to the downstream connected components.
        """
        self.type_ = Any
        component.set_input_types(self, value=Variadic[type_])
        # component.set_output_types(self, value=List[Dict[str,Any]])
        component.set_output_types(self, value=Dict[str,Any])

    def to_dict(self):
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, type_=serialize_type(self.type_))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Multiplexer":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
              Deserialized component.
        """
        data["init_parameters"]["type_"] = deserialize_type(data["init_parameters"]["type_"])
        
        return default_from_dict(cls, data)
    
    def run(self, **kwargs):
        """
        The value passed in must be a dictionary, otherwise an error will be reported
        
        """
        
        # print("kwargs\n", kwargs)
        if isinstance(kwargs["value"],list):
            value = kwargs["value"][-1]
            
            if isinstance(value,dict):
                value = value
            if isinstance(value,str):
                value = json.loads(value)
                
        if isinstance(kwargs["value"],dict):
            value = kwargs["value"]
        if isinstance(kwargs["value"],str):
            value = json.loads(kwargs["value"])
        
        if not isinstance(value, dict):
            raise ValueError(f"Multiplexer must be dict, but {value} is not.")

        print("----------Multiplexer-value------------")

        return {"value": value}
    