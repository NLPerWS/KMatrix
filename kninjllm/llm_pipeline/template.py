from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from jinja2 import Environment, PackageLoader, TemplateSyntaxError, meta

TEMPLATE_FILE_EXTENSION = ".yaml.jinja2"
TEMPLATE_HOME_DIR = Path(__file__).resolve().parent / "predefined"


class PredefinedPipeline(Enum):
    """
    Enumeration of predefined pipeline templates that can be used to create a `PipelineTemplate`.
    """

    # Maintain 1-to-1 mapping between the enum name and the template file name in templates directory
    GENERATIVE_QA = "generative_qa"
    RAG = "rag"
    INDEXING = "indexing"
    CHAT_WITH_WEBSITE = "chat_with_website"

class PipelineTemplate:


    def __init__(self, template_content: str):
        """
        Initialize a PipelineTemplate. Besides calling the constructor directly, a set of utility methods is provided
        for conveniently create an instance of `PipelineTemplate` from different sources. See `from_string`,
        `from_file`, `from_predefined` and `from_url`.

        :param template_content: The raw template source to use in the template.
        """
        env = Environment(
            loader=PackageLoader("kninjllm.llm_pipeline.pipeline", "predefined"), trim_blocks=True, lstrip_blocks=True
        )
        try:
            self._template = env.from_string(template_content)
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid pipeline template: {e.message}") from e

        # Store the list of undefined variables in the template. Components' names will be part of this list
        self.template_variables = meta.find_undeclared_variables(env.parse(template_content))
        self._template_content = template_content

    def render(self, template_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Constructs a `Pipeline` instance based on the template.

        :param template_params: An optional dictionary of parameters to use when rendering the pipeline template.

        :returns: An instance of `Pipeline` constructed from the rendered template and custom component configurations.
        """
        template_params = template_params or {}
        return self._template.render(**template_params)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> "PipelineTemplate":
        """
        Create a PipelineTemplate from a file.
        :param file_path: The path to the file containing the template. Must contain valid Jinja2 syntax.
        :returns: An instance of `PipelineTemplate`.
        """
        with open(file_path, "r") as file:
            return cls(file.read())

    @classmethod
    def from_predefined(cls, predefined_pipeline: PredefinedPipeline) -> "PipelineTemplate":
        """
        Create a PipelineTemplate from a predefined template. See `PredefinedPipeline` for available options.
        :param predefined_pipeline: The predefined pipeline to use.
        :returns: An instance of `PipelineTemplate `.
        """
        template_path = f"{TEMPLATE_HOME_DIR}/{predefined_pipeline.value}{TEMPLATE_FILE_EXTENSION}"
        return cls.from_file(template_path)

    @property
    def template_content(self) -> str:
        """
        Returns the raw template string as a read-only property.
        """
        return self._template_content
