import glob
import json
import os
from typing import Any, Dict, List, Optional

from kninjllm.llm_common.component import component
from kninjllm.llm_utils.common_utils import read_server_files
from root_config import RootConfig


@component
class DataSetLoader:

    def __init__(
        self,
        dataset_path: str
    ):
        self.dataset_path = dataset_path
        
        if self.dataset_path != "" and not self.dataset_path.startswith("/"):
            self.dataset_path = os.path.join(RootConfig.root_path,self.dataset_path)
        
        self.dataSetList = []
        
    @component.output_types(dataset_info=Dict[str, Any])
    def run(
        self
    ):
        
        if os.path.isfile(self.dataset_path):
            # self.dataSetList = read_server_files(self.dataset_path)[0:2]
            self.dataSetList = read_server_files(self.dataset_path)
        
        return {"dataset_info": {"dataset_path":self.dataset_path,"dataset_data":self.dataSetList }}