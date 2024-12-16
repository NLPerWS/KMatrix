from kninjllm.llm_common.component import component

@component
class KnowledgeUploader:
    def __init__(
        self,
        path
    ):
        self.path = path

    @component.output_types(path=str)
    def run(self):
        print("------------------KnowledgeUploader----------------------")
        return {"path":self.path}