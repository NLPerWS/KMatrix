


"""
    Attributes:

        component: Marks a class as a component. Any class decorated with `@component` can be used by a Pipeline.

    All components must follow the contract below. This docstring is the source of truth for components contract.

    <hr>

    `@component` decorator

    All component classes must be decorated with the `@component` decorator. This allows Canals to discover them.

    <hr>

    `__init__(self, **kwargs)`

    Optional method.

    Components may have an `__init__` method where they define:

    - `self.init_parameters = {same parameters that the __init__ method received}`:
        In this dictionary you can store any state the components wish to be persisted when they are saved.
        These values will be given to the `__init__` method of a new instance when the pipeline is loaded.
        Note that by default the `@component` decorator saves the arguments automatically.
        However, if a component sets their own `init_parameters` manually in `__init__()`, that will be used instead.
        Note: all of the values contained here **must be JSON serializable**. Serialize them manually if needed.

    Components should take only "basic" Python types as parameters of their `__init__` function, or iterables and
    dictionaries containing only such values. Anything else (objects, functions, etc) will raise an exception at init
    time. If there's the need for such values, consider serializing them to a string.

    _(TODO explain how to use classes and functions in init. In the meantime see `test/components/test_accumulate.py`)_

    The `__init__` must be extremely lightweight, because it's a frequent operation during the construction and
    validation of the pipeline. If a component has some heavy state to initialize (models, backends, etc...) refer to
    the `warm_up()` method.

    <hr>

    `warm_up(self)`

    Optional method.

    This method is called by Pipeline before the graph execution. Make sure to avoid double-initializations,
    because Pipeline will not keep track of which components it called `warm_up()` on.

    <hr>

    `run(self, data)`

    Mandatory method.

    This is the method where the main functionality of the component should be carried out. It's called by
    `Pipeline.run()`.

    When the component should run, Pipeline will call this method with an instance of the dataclass returned by the
    method decorated with `@component.input`. This dataclass contains:

    - all the input values coming from other components connected to it,
    - if any is missing, the corresponding value defined in `self.defaults`, if it exists.

    `run()` must return a single instance of the dataclass declared through the method decorated with
    `@component.output`.

"""

import inspect
import sys
from collections.abc import Callable
from copy import deepcopy
from types import new_class
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from kninjllm.llm_common.errors import ComponentError
from kninjllm.llm_common.sockets import Sockets
from kninjllm.llm_common.types import InputSocket, OutputSocket, _empty


@runtime_checkable
class Component(Protocol):
    """
    Note this is only used by type checking tools.

    In order to implement the `Component` protocol, custom components need to
    have a `run` method. The signature of the method and its return value
    won't be checked, i.e. classes with the following methods:

        def run(self, param: str) -> Dict[str, Any]:
            ...

    and

        def run(self, **kwargs):
            ...

    will be both considered as respecting the protocol. This makes the type
    checking much weaker, but we have other places where we ensure code is
    dealing with actual Components.

    The protocol is runtime checkable so it'll be possible to assert:

        isinstance(MyComponent, Component)
    """

    
    
    
    
    
    if sys.version_info >= (3, 9):
        run: Callable[..., Dict[str, Any]]
    else:
        run: Callable


class ComponentMeta(type):
    def __call__(cls, *args, **kwargs):
        """
        This method is called when clients instantiate a Component and
        runs before __new__ and __init__.
        """
        
        instance = super().__call__(*args, **kwargs)

        
        

        
        
        if not hasattr(instance, "__kninjllm_output__"):
            
            
            
            
            
            
            instance.__kninjllm_output__ = Sockets(
                instance, deepcopy(getattr(instance.run, "_output_types_cache", {})), OutputSocket
            )

        
        
        if not hasattr(instance, "__kninjllm_input__"):
            instance.__kninjllm_input__ = Sockets(instance, {}, InputSocket)
        run_signature = inspect.signature(getattr(cls, "run"))
        for param in list(run_signature.parameters)[1:]:  
            if run_signature.parameters[param].kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):  
                socket_kwargs = {"name": param, "type": run_signature.parameters[param].annotation}
                if run_signature.parameters[param].default != inspect.Parameter.empty:
                    socket_kwargs["default_value"] = run_signature.parameters[param].default
                instance.__kninjllm_input__[param] = InputSocket(**socket_kwargs)

        
        
        
        instance.__kninjllm_added_to_pipeline__ = None

        
        
        
        is_variadic = any(socket.is_variadic for socket in instance.__kninjllm_input__._sockets_dict.values())

        return instance


def _component_repr(component: Component) -> str:
    """
    All Components override their __repr__ method with this one.
    It prints the component name and the input/output sockets.
    """
    result = object.__repr__(component)
    if pipeline := getattr(component, "__kninjllm_added_to_pipeline__"):
        
        result += f"\n{pipeline.get_component_name(component)}"

    
    
    return f"{result}\n{component.__kninjllm_input__}\n{component.__kninjllm_output__}"  


class _Component:
    """
    See module's docstring.

    Args:
        class_: the class that Canals should use as a component.
        serializable: whether to check, at init time, if the component can be saved with
        `save_pipelines()`.

    Returns:
        A class that can be recognized as a component.

    Raises:
        ComponentError: if the class provided has no `run()` method or otherwise doesn't respect the component contract.
    """

    def __init__(self):
        self.registry = {}

    def set_input_type(self, instance, name: str, type: Any, default: Any = _empty):
        """
        Add a single input socket to the component instance.

        :param instance: Component instance where the input type will be added.
        :param name: name of the input socket.
        :param type: type of the input socket.
        :param default: default value of the input socket, defaults to _empty
        """
        if not hasattr(instance, "__kninjllm_input__"):
            instance.__kninjllm_input__ = Sockets(instance, {}, InputSocket)
        instance.__kninjllm_input__[name] = InputSocket(name=name, type=type, default_value=default)

    def set_input_types(self, instance, **types):
        """
        Method that specifies the input types when 'kwargs' is passed to the run method.

        Use as:

        ```python
        @component
        class MyComponent:

            def __init__(self, value: int):
                component.set_input_types(value_1=str, value_2=str)
                ...

            @component.output_types(output_1=int, output_2=str)
            def run(self, **kwargs):
                return {"output_1": kwargs["value_1"], "output_2": ""}
        ```

        Note that if the `run()` method also specifies some parameters, those will take precedence.

        For example:

        ```python
        @component
        class MyComponent:

            def __init__(self, value: int):
                component.set_input_types(value_1=str, value_2=str)
                ...

            @component.output_types(output_1=int, output_2=str)
            def run(self, value_0: str, value_1: Optional[str] = None, **kwargs):
                return {"output_1": kwargs["value_1"], "output_2": ""}
        ```

        would add a mandatory `value_0` parameters, make the `value_1`
        parameter optional with a default None, and keep the `value_2`
        parameter mandatory as specified in `set_input_types`.

        """
        instance.__kninjllm_input__ = Sockets(
            instance, {name: InputSocket(name=name, type=type_) for name, type_ in types.items()}, InputSocket
        )

    def set_output_types(self, instance, **types):
        """
        Method that specifies the output types when the 'run' method is not decorated
        with 'component.output_types'.

        Use as:

        ```python
        @component
        class MyComponent:

            def __init__(self, value: int):
                component.set_output_types(output_1=int, output_2=str)
                ...

            
            def run(self, value: int):
                return {"output_1": 1, "output_2": "2"}
        ```
        """
        instance.__kninjllm_output__ = Sockets(
            instance, {name: OutputSocket(name=name, type=type_) for name, type_ in types.items()}, OutputSocket
        )

    def output_types(self, **types):
        """
        Decorator factory that specifies the output types of a component.

        Use as:

        ```python
        @component
        class MyComponent:
            @component.output_types(output_1=int, output_2=str)
            def run(self, value: int):
                return {"output_1": 1, "output_2": "2"}
        ```
        """

        def output_types_decorator(run_method):
            """
            This happens at class creation time, and since we don't have the decorated
            class available here, we temporarily store the output types as an attribute of
            the decorated method. The ComponentMeta metaclass will use this data to create
            sockets at instance creation time.
            """
            setattr(
                run_method,
                "_output_types_cache",
                {name: OutputSocket(name=name, type=type_) for name, type_ in types.items()},
            )
            return run_method

        return output_types_decorator

    def _component(self, cls, is_greedy: bool = False):
        """
        Decorator validating the structure of the component and registering it in the components registry.
        """
        
        if not hasattr(cls, "run"):
            raise ComponentError(f"{cls.__name__} must have a 'run()' method. See the docs for more information.")

        def copy_class_namespace(namespace):
            """
            This is the callback that `typing.new_class` will use
            to populate the newly created class. We just copy
            the whole namespace from the decorated class.
            """
            for key, val in dict(cls.__dict__).items():
                
                if key in ("__dict__", "__weakref__"):
                    continue
                namespace[key] = val

        
        
        
        
        cls: cls.__name__ = new_class(cls.__name__, cls.__bases__, {"metaclass": ComponentMeta}, copy_class_namespace)  

        
        class_path = f"{cls.__module__}.{cls.__name__}"
        self.registry[class_path] = cls

        
        cls.__repr__ = _component_repr

        
        
        
        
        setattr(cls, "__kninjllm_is_greedy__", is_greedy)

        return cls

    def __call__(self, cls: Optional[type] = None, is_greedy: bool = False):
        
        
        def wrap(cls):
            return self._component(cls, is_greedy=is_greedy)

        if cls:
            
            return wrap(cls)

        
        return wrap


component = _Component()
