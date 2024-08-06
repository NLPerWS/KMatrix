
export function hasLine(data, from, to) {
  for (let i = 0; i < data.lineList.length; i++) {
    let line = data.lineList[i];
    if (line.from === from && line.to === to) {
      return true;
    }
  }
  return false;
}


export function hashOppositeLine(data, from, to) {
  return hasLine(data, to, from);
}


export function getConnector(jsp, from, to) {
  let connection = jsp.getConnections({
    source: from,
    target: to
  })[0];
  return connection;
}


export function uuid() {
  return Math.random()
    .toString(36)
    .substr(3, 10);
}

import { rootUrl } from "@/assets/js/host";
import "@/assets/js/jquery371";


export function do_getInitConfigData(thisNode) {
  
  let that = thisNode;
  let result = null;
  $.ajax({
    type: "get",
    dataType: "json",
    contentType: "application/json",
    async: false,
    url: rootUrl + "/getInitConfig",
    success: function(data) {
      if (data.code == 200) {
        result = data.data;
      } else {
        that.$message({ message: data.data, type: "error", duration: 3000 });
        result = data.data;
      }
    },
    error: function(err) {
      that.$message({
        message: "Network exception, please try again later",
        type: "error",
        duration: 2000
      });
      result = "Network exception, please try again later";
    }
  });
  return result;
}


export function do_saveConfigData(
  thisNode,
  ConfigDataJson,
  action,
  updateNode
) {
  
  let that = thisNode;
  $.ajax({
    type: "post",
    dataType: "json",
    contentType: "application/json",
    async: false,
    url: rootUrl + "/uploadComponentConfig",
    data: JSON.stringify({
      data: ConfigDataJson,
      action: action,
      updateNode: updateNode
    }),
    success: function(data) {
      if (data.code == 200) {
        that.$message({ message: "operation is successful", type: "success", duration: 2000 });
      } else {
        that.$message({ message: data.data, type: "error", duration: 2000 });
        return;
      }
    },
    error: function(err) {
      that.$message({
        message: "Network exception, please try again later",
        type: "error",
        duration: 2000
      });
      return;
    }
  });
}


export function do_getComponentCodeByFilePath(thisNode, codeFilePath) {
  let that = thisNode;
  let result = null;
  $.ajax({
    type: "post",
    dataType: "json",
    contentType: "application/json",
    async: false,
    url: rootUrl + "/getComponentCodeByFilePath",
    data: JSON.stringify({ codeFilePath: codeFilePath }),
    success: function(data) {
      if (data.code == 200) {
        
        result = data.data;
      } else {
        that.$message({ message: data.data, type: "error", duration: 2000 });
        result = data.data;
      }
    },
    error: function(err) {
      that.$message({
        message: "Network exception, please try again later",
        type: "error",
        duration: 2000
      });
      result = "Network exception, please try again later";
    }
  });
  return result;
}


export function do_setComponentCodeByFilePath(thisNode, codeFilePath, code) {
  let that = thisNode;
  
  $.ajax({
    type: "post",
    dataType: "json",
    contentType: "application/json",
    async: false,
    url: rootUrl + "/setComponentCodeByFilePath",
    data: JSON.stringify({ codeFilePath: codeFilePath, code: code }),
    success: function(data) {
      if (data.code == 200) {
        that.$message({ message: "operation is successful", type: "success", duration: 2000 });
        
      } else {
        that.$message({ message: data.data, type: "error", duration: 2000 });
        return;
      }
    },
    error: function(err) {
      that.$message({
        message: "Network exception, please try again later",
        type: "error",
        duration: 2000
      });
      return;
    }
  });
  
}


export function do_getAllJsonData(thisNode) {

  let that = thisNode;
  let result = null;
  $.ajax({
    type: "get",
    dataType: "json",
    contentType: "application/json",
    async: false,
    url: rootUrl + "/getAllJsonData",
    success: function(data) {
      
      if (data.code == 200) {
        result = data["data"];
      } else {
        that.$message({
          message: "Service exception, please try again later",
          type: "error",
          duration: 1500
        });
        result = data.data;
      }
    },
    error: function(err) {
      that.$message({
        message: "Network exception, please try again later",
        type: "error",
        duration: 1500
      });
      result = "Network exception, please try again later";
    }
  });
  return result;
}


export function do_getResultData(thisNode) {
  if (
    thisNode.data.name == "undefined" ||
    thisNode.data.name == undefined ||
    thisNode.data.name == ""
  ) {
    thisNode.$message({
      message: "Please select a diagram first",
      type: "warning",
      duration: 1500
    });
    return;
  }
  let that = thisNode;
  $.ajax({
    type: "post",
    dataType: "json",
    contentType: "application/json",
    async: true,
    url: rootUrl + "/getPipelineResultData",
    data: JSON.stringify({ pipeLineName: thisNode.data.name,pipeLineType:thisNode.data.pipeLineType }),
    success: function(data) {
      
      if (data.code == 200) {
        that.resultData = data["data"];
        that.flowInfoVisible = true;
        that.$nextTick(function() {
          that.$refs.flowInfo.init();
        });
      } else {
        that.$message({ message: data.data, type: "warning", duration: 1500 });
      }
    },
    error: function(err) {
      that.$message({
        message: "Network exception, please try again later",
        type: "error",
        duration: 1500
      });
      return;
    }
  });
}


export function do_getVariableData(thisNode, thisPipeineName,pipeLineType) {
  let that = thisNode;
  console.log("🚀 -> pipeLineType:\n", pipeLineType)
  let result = [];
  $.ajax({
    type: "post",
    dataType: "json",
    contentType: "application/json",
    async: false,
    url: rootUrl + "/getPipelineTempVariableData",
    data: JSON.stringify({ pipeLineName: thisPipeineName,pipeLineType:pipeLineType }),
    success: function(data) {
      
      if (data.code == 200) {
        result = data["data"];
      } else {
        that.$message({ message: data.data, type: "error", duration: 1500 });
      }
    },
    error: function(err) {
      that.$message({
        message: "Network exception, please try again later",
        type: "error",
        duration: 1500
      });
    }
  });

  return result;
}


export function do_delPipelineByFileName(thisNode, pipeLineName,pipeLineType) {
  let that = thisNode;
  $.ajax({
    type: "post",
    dataType: "json",
    contentType: "application/json",
    async: true,
    url: rootUrl + "/delPipelineByFileName",
    data: JSON.stringify({ pipeLineName: pipeLineName,pipeLineType:pipeLineType}),
    success: function(data) {
      if (data.code == 200) {
        that.$message({ message: data.data, type: "success", duration: 2000 });
      } else {
        that.$message({ message: data.data, type: "error", duration: 2000 });
        return;
      }
    },
    error: function(err) {
      that.$message({
        message: "Network exception, please try again later",
        type: "error",
        duration: 2000
      });
      return;
    }
  });
}


export function do_savePipeline(thisNode) {
  let that = thisNode;
  $.ajax({
    type: "post",
    dataType: "json",
    contentType: "application/json",
    async: true,
    url: rootUrl + "/startRunPipelineByFileName",
    data: JSON.stringify({
      data: thisNode.data,
      pipeLineName: thisNode.data.name,
      execute: "save"
    }),
    success: function(data) {
      
      if (data.code == 200) {
        that.$message({ message: data.data, type: "success", duration: 2000 });
      } else {
        that.$message({ message: data.data, type: "error", duration: 2000 });
        return;
      }
    },
    error: function(err) {
      that.$message({
        message: "Network exception, please try again later",
        type: "error",
        duration: 2000
      });
      return;
    }
  });
}


export function do_initCatch() {
  $.ajax({
    type: "get",
    dataType: "json",
    contentType: "application/json",
    async: false,
    url: rootUrl + "/initCatch",
    success: function(data) {

    },
    error: function(err) {
      that.$message({
        message: "Network exception, please try again later",
        type: "error",
        duration: 2000
      });
      return;
    }
  });
  return;
}


export function do_executePipeline(thisNode,message,mode) {
  let that = thisNode;
  let execute = "run";
  if(mode != undefined && mode != null){
    execute = mode;
  }
  $.ajax({
    type: "post",
    dataType: "json",
    contentType: "application/json",
    async: true,
    url: rootUrl + "/startRunPipelineByFileName",
    data: JSON.stringify({
      data: thisNode.data,
      pipeLineName: thisNode.data.name,
      execute: execute
    }),
    success: function(data) {
      
      if (data.code == 200) {
        that.$message({ message: "executes successfully", type: "success", duration: 2000 });
        if (message != undefined && message!=null){
          message.close();
        }
        if (mode != undefined){
          that.QAModel = true;
          that.pageModel = "流图部署";
        }
      } else {
        that.$message({ message: data.data, type: "error", duration: 3000 });
        if (message != undefined && message!=null){
          message.close();
        }
        return;
      }
    },
    error: function(err) {
      that.$message({
        message: "Network exception, please try again later",
        type: "error",
        duration: 2000
      });
      if (message != undefined && message!=null){
        message.close();
      }
      return;
    }
  });
}


function extractUpToSecondArrow(str) {
  
  let firstIndex = str.indexOf('->');
  
  if (firstIndex !== -1) {
    let secondIndex = str.indexOf('->', firstIndex + 2);
    
    if (secondIndex !== -1) {
      return str.substring(0, secondIndex + 2);
    }
  }
  
  return "";
}



function processStringArray(arr) {
  let countMap = {};
  return arr.map((str) => {
      if (str.includes('Function ->')){
        let checkStr = extractUpToSecondArrow(str);

        if ( checkStr.includes('TableRetriever') || checkStr.includes('SqlRetriever') || checkStr.includes('KGRetriever') ) {
            countMap[checkStr] = (countMap[checkStr] || 0) + 1;
            
            return `<strong style="font-size:1.1rem;">Query Count: ${countMap[checkStr]}</strong> ${str}`;
        }
  
        else if (checkStr.includes('Retriever')) {
            countMap[checkStr] = (countMap[checkStr] || 0) + 1;
            
            return `<strong style="font-size:1.1rem;">Retriever Count: ${countMap[checkStr]}</strong> ${str}`;
        }
        else if (checkStr.includes('Generator')) {
            countMap[checkStr] = (countMap[checkStr] || 0) + 1;
            
            return `<strong style="font-size:1.1rem;">Generate Count: ${countMap[checkStr]}</strong> ${str}`;
        }
        else{
          
        }
      }
      return str;
  });
}


export function do_executePipeline_QA(thisNode) {

  let that = thisNode;
  $.ajax({
    type: "post",
    dataType: "json",
    contentType: "application/json",
    async: true,
    url: rootUrl + "/startRunPipelineByFileName",
    data: JSON.stringify({
      data: thisNode.data,
      pipeLineName: thisNode.data.name,
      execute: "Deployment"
    }),
    success: function(data) {
      
      if (data.code == 200) {
        that.$message({ message: "executes successfully", type: "success", duration: 2000 });
        let result = data.data;
        let resultDataContent = "";
        let resultDataLogs = "";
        try {
          resultDataContent = result["OutputBuilder"]["final_result"]["content"];
          resultDataLogs =  result["OutputBuilder"]["final_result"]["ctxs"];
        } catch (error) {
          console.log(
            "result error (must have result['OutputBuilder']['final_result']['content'] 与 ['ctxs']) \n", result );
        }
        
        that.AnswerData = resultDataContent;
        resultDataLogs = processStringArray(resultDataLogs);
        that.RetrieverData = resultDataLogs.map(e => {
          return { text: e };
        });
        that.infer_loading = false;
      } else {
        that.$message({ message: data.data, type: "error", duration: 3000 });
        that.infer_loading = false;
        return;
      }
    },
    error: function(err) {
      that.$message({
        message: "Network exception, please try again later",
        type: "error",
        duration: 2000
      });
      that.infer_loading = false;
      return;
    }
  });


}
