<template>
  <div  v-if="easyFlowVisible" class="root">

    <el-row>
      <!--顶部工具菜单-->
      <el-col :span="24">
        <div class="ef-tooltar" >
          <div class="title-logo"></div>

          <div class="title-edit">
            <el-tooltip :content="$t('topmenu.edit_pipeline_name')" placement="top">
              <el-link
                :underline="false"
                @click="dialogVisible_input_name = true"
                >{{ data.name }}</el-link
              >
            </el-tooltip>

            <el-dialog
              :title="$t('topmenu.edit_pipeline_name_title')"
              :visible.sync="dialogVisible_input_name"
              width="30%"
            >
              <el-input v-model="data.name" placeholder="" @input="handleInput"></el-input>
              <span slot="footer" class="dialog-footer">
                <el-button  class="my-el-button" @click="dialogVisible_input_name = false"
                  >{{$t("common.return")}}</el-button
                >
                <el-button
                  class="my-el-button" 
                  @click="dialogVisible_input_name = false"
                  >{{$t("common.ok")}}</el-button
                >
              </span>
            </el-dialog>

            <el-divider direction="vertical"></el-divider>

            <el-select v-model="pipeLineType" :placeholder="$t('topmenu.select_pipeline_type')">
              <el-option
                v-for="item in pipeLineTypeOptions"
                :key="item"
                :label="item"
                :value="item"
              >
              </el-option>
            </el-select>

            <el-divider direction="vertical"></el-divider>
            <el-tooltip :content="$t('topmenu.create_common_com')" placement="top">
              <el-button
                type="text"
                icon="el-icon-edit"
                size="large"
                @click="dialogVisible_createElement = true"
              ></el-button>
            </el-tooltip>

            <el-dialog
              :title="$t('topmenu.create_common_com_title')"
              :visible.sync="dialogVisible_createElement"
              :width="dialog_width"
            >
              <el-form :model="newComponent">
                <el-form-item :label="$t('topmenu.create_common_com_parent')">
                  <el-select
                    v-model="newComponent.parent_type"
                  >
                    <el-option
                      v-for="(item, index) in config_parent_list"
                      :key="item.show"
                      :label="item.show"
                      :value="index"
                    >
                    </el-option>
                  </el-select>
                </el-form-item>

                <el-form-item :label="$t('topmenu.create_common_com_name')">
                  <el-input
                    v-model="newComponent.name"
                    placeholder=""
                    style="width: 400px;"
                  >
                  </el-input>
                </el-form-item>

                <el-form-item :label="$t('topmenu.create_common_com_path')">
                  <div id="newComponent_filePath" >
                    <span >kninjllm/</span>
                    <span v-text="config_parent_list[newComponent.parent_type].path"></span>
                    <span>/</span>
                    <span v-text="newComponent.name"></span>
                    <span >.py</span>
                  </div>
                </el-form-item>

              </el-form>

              <span slot="footer" class="dialog-footer">
                <el-button @click="dialogVisible_createElement = false"  class="my-el-button" 
                  > {{ $t('common.return') }} </el-button
                >
                <el-button  @click="createCommonElement"  class="my-el-button" 
                  > {{ $t('common.ok')  }} </el-button
                >
              </span>
            </el-dialog>

            <el-divider direction="vertical"></el-divider>
            <el-tooltip :content="$t('topmenu.create_control_com')" placement="top">
              <el-button
                type="text"
                icon="el-icon-cpu"
                size="large"
                @click="dialogVisible_createControllElement = true"
              ></el-button>
            </el-tooltip>

            <el-dialog
              :title="$t('topmenu.create_control_com_title')"
              :visible.sync="dialogVisible_createControllElement"
              :width="dialog_width"
            >
              <el-form :model="controllComponent">
                <el-form-item :label="$t('topmenu.create_control_com_coms')">
                  <el-select
                    @change="changeCommonCompent"
                    v-model="controllComponent.children_componentList"
                    filterable
                    allow-create
                    multiple
                    default-first-option
                  >
                    <el-option
                      v-for="item in config_children_list"
                      :key="item.type"
                      :label="item.type"
                      :value="item.type"
                    >
                    </el-option>
                  </el-select>
                </el-form-item>

                <el-form-item :label="$t('topmenu.create_control_com_name')">
                  <el-input
                    v-model="controllComponent.name"
                    placeholder=""
                    style="width: 400px;"
                  >
                    <template slot="append">Controller</template>
                  </el-input>
                </el-form-item>

                <el-form-item :label="$t('topmenu.create_control_com_path')">

                  <div id="controllerComponent_filePath">
                    <span >kninjllm/llm_controller/</span>
                    <span v-text="controllComponent.name"></span>
                    <span >Controller</span>
                    <span >.py</span>
                  </div>

                </el-form-item>

                <el-form-item :label="$t('topmenu.create_control_com_init')">
                  <textarea
                    class="edit textarea-param" 
                    rows="5"
                    v-model="controllComponent.initParams"
                  ></textarea>
                </el-form-item>

                <el-form-item :label="$t('topmenu.create_control_com_input')">
                  <textarea
                    class="edit  textarea-param"
                    rows="5"
                    v-model="controllComponent.inputParams"
                  ></textarea>
                </el-form-item>

                <el-form-item :label="$t('topmenu.create_control_com_output')">
                  <textarea
                    class="edit  textarea-param"
                    rows="5"
                    v-model="controllComponent.outputParams"
                  ></textarea>
                </el-form-item>
              </el-form>

              <span slot="footer" class="dialog-footer">
                <el-button class="my-el-button" @click="dialogVisible_createControllElement = false"
                  >{{ $t('common.return') }}</el-button
                >
                <el-button class="my-el-button"  @click="createControllElement"
                  > {{ $t('common.ok') }} </el-button
                >
              </span>
            </el-dialog>

            <el-divider direction="vertical"></el-divider>
            <el-tooltip :content="$t('topmenu.del_pipeline_com')" placement="top" >
              <el-button
                type="text"
                icon="el-icon-delete"
                size="large"
                @click="deleteElement"
                :disabled="!this.activeElement.type"
              ></el-button>
            </el-tooltip>

          </div>

          <div class="title-right">

            <el-cascader v-if="QAModel == false "
              v-model="thisPipeline"
              :placeholder="$t('topmenu.use_pipeline')"
              @change="selectPipeline"
              @focus="setPipelineData"
              :options="pipelineList"
            >
              <template slot-scope="{ node, data }">
                <span> {{ data.value }} </span>
                <el-button v-if="node.isLeaf"  class="my-del_el-button" @click.stop="handleDeletePipeline(data, $event)">{{ $t('topmenu.do_del') }}</el-button>
              </template>
            </el-cascader>

            <el-button class="my-el-button"
              v-if="pipeLineType==='Deployment'  && QAModel == false "
              round
              @click="dev_pipelline()"
              icon="el-icon-news"
              size="mini"
              >{{ $t('topmenu.dev_pipeline') }}</el-button
            >

            <el-button class="my-el-button"
              v-if="pipeLineType==='Deployment' && QAModel == true " 
              round
              @click="return_design_page()"
              icon="el-icon-news"
              size="mini"
              > {{ $t('topmenu.return_design_page') }} </el-button
            >

            <el-button v-if="QAModel == false && pipeLineType!='Deployment'" class="my-el-button"
              round
              @click="runPipeline"
              icon="el-icon-position"
              size="mini"
              >{{  $t("topmenu.execute_pipeline")  }}</el-button
            >

            <el-button v-if="QAModel == false" class="my-el-button"
              round
              @click="savePipeline"
              icon="el-icon-takeaway-box"
              size="mini" 
              >{{ $t("topmenu.save_pipeline") }} </el-button
            >


            <el-button v-if="QAModel == false" class="my-el-button"
              round
              icon="el-icon-document"
              @click="getResultData"
              size="mini"
              >{{ $t("topmenu.check_pipeline") }}</el-button
            >

            <el-button v-if="QAModel == false" class="my-el-button"
              round
              @click="createPipe"
              icon="el-icon-news"
              size="mini"
              > {{ $t("topmenu.return") }} </el-button
            >

            <el-select style="width: 100px;" v-model="languageValue" @change="changeLanguage" placeholder="">
                  <el-option
                    v-for="item in languageOptions"
                    :key="item.value"
                    :label="item.label"
                    :value="item.value">
                  </el-option>
            </el-select>

          </div>

        </div>
      </el-col>
    </el-row>


      <div v-if="QAModel==true" class='page'>
        
        <QA :data="this.data" :pipeLineType="this.pipeLineType" :pipeline_components=this.pipeline_components :pipeline_info=this.pipeline_info></QA>

      </div>

      <div v-else class='page'>

          <!--左侧菜单 -->
          <div class="left_div" >
            <node-menu @addNode="addNode" ref="nodeMenu"></node-menu>
          </div>

          <!-- 中间画布 -->
          <div id="efContainer" ref="efContainer" class="container" v-flowDrag>
            <template v-for="node in data.nodeList">
              <flow-node
                :id="node.id"
                :node="node"
                :activeElement="activeElement"
                @contextmenu.prevent="onContextmenu"
                @changeNodeSite="changeNodeSite"
                @nodeRightMenu="nodeRightMenu"
                @clickNode="clickNode"
              >
              </flow-node>
            </template>
            <!-- 给画布一个默认的宽度和高度 -->
            <div style="position:absolute;top: 2000px;left: 2000px; ">&nbsp;</div>
          </div>

          <!-- 右侧表单 -->
          <div class="right_div">
            <flow-node-form
              :pipelineName="this.data.name"
              :pipelineType="this.data.pipeLineType"
              ref="nodeForm"
              @setLineLabel="setLineLabel"
              @repaintEverything="repaintEverything"
            >
            </flow-node-form>
          </div>
      
          <!-- 流程数据详情 -->
          <flow-info
            v-if="flowInfoVisible"
            ref="flowInfo"
            :data="resultData"
          >
          </flow-info>
          <flow-help v-if="flowHelpVisible" ref="flowHelp"></flow-help>
      </div>

    </div>

</template>

<script>
import draggable from "vuedraggable";
// import { jsPlumb } from 'jsplumb'
import "@/assets/js/jsplumb";
import { easyFlowMixin } from "@/assets/js/mixins";
import flowNode from "@/components/ef/node";
import nodeMenu from "@/components/ef/node_menu";
import FlowInfo from "@/components/ef/info";
import FlowHelp from "@/components/ef/help";
import FlowNodeForm from "@/components/ef/node_form";
import QA from "@/components/ef/QA_v2";

import lodash from "lodash";
import { ForceDirected } from "@/assets/js/force-directed";
import * as utils from "@/assets/js/utils";

var initConfigDataJson = [];

export default {
  data() {
    return {
      languageValue:'',
      languageOptions:[],
      jsPlumb: null,
      easyFlowVisible: true,
      flowInfoVisible: false,
      loadEasyFlowFinish: false,
      flowHelpVisible: false,
      data: {},
      activeElement: {
        type: undefined,
        nodeId: undefined,
        sourceId: undefined,
        targetId: undefined
      },
      zoom: 0.5,
      dialogVisible_input_name: false,
      pipeLineTypeOptions: [],
      pipeLineType: "",
      pipelineList: [],
      thisPipeline: "",
      resultData: {},
      dialog_width: "90%",
      dialogVisible_createElement: false,
      dialogVisible_createControllElement: false,
      config_parent_list: [
        {"path":"llm_generator","type":"Generator","show":"Generator"},
        {"path":"llm_retriever","type":"Retriever","show":"Retriever"}
      ],
      config_children_list: [],
      newComponent: {
        parent_type: 0,
        type: "",
        initParams: '[{"name": "","type": "","value": ""}]',
        inputParams: '[{"name": "","type": "","value": ""}]',
        outputParams: '[{"name": "","type": "","value": ""}]',
        codeFilePath: "",
        description: "",
      },
      
      controllComponent: {
        type: "",
        children_componentList: [],
        initParams: '[{"name": "","type": "","value": ""}]',
        tempControllerValueConfig:[],
        inputParams: '[{"name": "","type": "","value": ""}]',
        outputParams: '[{"name": "","type": "","value": ""}]',
        codeFilePath: "",
        description: "",
      },
      // 部署 QA页面
      QAModel: false,
      pipeline_components:[],
      pipeline_info:[],
      pageModel:"流图设计"
    };
  },
  // 一些基础配置移动该文件中
  mixins: [easyFlowMixin],
  components: {
    draggable,
    flowNode,
    nodeMenu,
    FlowInfo,
    FlowNodeForm,
    FlowHelp,
    QA
  },
  directives: {
    flowDrag: {
      bind(el, binding, vnode, oldNode) {
        if (!binding) {
          return;
        }
        el.onmousedown = e => {
          if (e.button == 2) {
            return;
          }
          let disX = e.clientX;
          let disY = e.clientY;
          el.style.cursor = "move";

          document.onmousemove = function(e) {
            // 移动时禁止默认事件
            e.preventDefault();
            const left = e.clientX - disX;
            disX = e.clientX;
            el.scrollLeft += -left;

            const top = e.clientY - disY;
            disY = e.clientY;
            el.scrollTop += -top;
          };

          document.onmouseup = function(e) {
            el.style.cursor = "auto";
            document.onmousemove = null;
            document.onmouseup = null;
          };
        };
      }
    }
  },

	created() {
		//最开始请求的时候看缓存是什么状态
		if(this.$i18n.locale=='zh'){
	      this.languageValue='中文';
	      this.languageOptions=[{value:'en',label:'English'}]
	    }else{
	      this.languageValue='English';
	      this.languageOptions=[{value:'zh',label:'中文'}]
	    }
	},
  mounted() {
    this.setPipelineData();
    this.updateConfig();
    this.jsPlumb = jsPlumb.getInstance();
    this.$nextTick(() => {
      this.createPipe();
    });
  },
  methods: {

    // 多语言切换
    changeLanguage(type){
      console.log(type);
      localStorage.setItem('locale',type)
      this.$i18n.locale = type; 
      if(this.$i18n.locale=='zh'){
        this.languageValue='中文';
        this.languageOptions=[{value:'en',label:'English'}]
      }else{
        this.languageValue='English';
        this.languageOptions=[{value:'zh',label:'中文'}]
      }
    },

    handleClose(done) {
      this.$confirm("确认关闭？")
        .then(_ => {
          done();
        })
        .catch(_ => {});
    },

    updateValue(index, indexparam, value) {
      this.$set(this.controllComponent.tempControllerValueConfig[index].initParams, indexparam, JSON.parse(value));
    },

    // 更新左侧配置
    updateConfig() {
      initConfigDataJson = [].concat(utils.do_getInitConfigData());
      for (let i in initConfigDataJson) {
        if (initConfigDataJson[i]["type"] != "Controller") {
          for (let j in initConfigDataJson[i]["children"]) {
            this.config_children_list.push(
              initConfigDataJson[i]["children"][j]
            );
          }
        }
      }
      console.log("初始化完成");
    },

    // 返回唯一标识
    getUUID() {
      return Math.random()
        .toString(36)
        .substr(3, 10);
    },
    jsPlumbInit() {
      this.jsPlumb.ready(() => {
        // 导入默认配置
        this.jsPlumb.importDefaults(this.jsplumbSetting);
        // 会使整个jsPlumb立即重绘。
        this.jsPlumb.setSuspendDrawing(false, true);
        // 初始化节点
        this.loadEasyFlow();
        // 单点击了连接线, https://www.cnblogs.com/ysx215/p/7615677.html
        this.jsPlumb.bind("click", (conn, originalEvent) => {
          this.activeElement.type = "line";
          this.activeElement.sourceId = conn.sourceId;
          this.activeElement.targetId = conn.targetId;

          let fromNode = this.data.nodeList.find(
            node => node.id === conn.sourceId
          );
          let toNode = this.data.nodeList.find(
            node => node.id === conn.targetId
          );
          let fromNodeOutputNameList = fromNode.outputParams.map(
            item => item.name
          );
          let toNodeInputNameList = toNode.inputParams.map(item => item.name);
          console.log("🚀 -> 3");
          this.$refs.nodeForm.lineInit({
            from: conn.sourceId,
            to: conn.targetId,
            label: conn.getLabel(),
            label_from_outputList: fromNodeOutputNameList,
            label_to_inputList: toNodeInputNameList,
            label_from: conn.getLabel().split(" -> ")[0],
            label_to: conn.getLabel().split(" -> ")[1]
          });
        });

        // 连线
        this.jsPlumb.bind("connection", evt => {
          let from = evt.source.id;
          let to = evt.target.id;
          let fromNode = this.data.nodeList.find(node => node.id === from);
          let toNode = this.data.nodeList.find(node => node.id === to);
          let fromNodeOutputNameList = fromNode.outputParams.map(
            item => item.name
          );
          let toNodeInputNameList = toNode.inputParams.map(item => item.name);
          if (this.loadEasyFlowFinish) {
            this.data.lineList.push({
              from: from,
              to: to,
              label_from: "",
              label_to: "",
              label_from_outputList: fromNodeOutputNameList,
              label_to_inputList: toNodeInputNameList
            });
          }
        });

        // 删除连线回调
        this.jsPlumb.bind("connectionDetached", evt => {
          this.deleteLine(evt.sourceId, evt.targetId);
        });

        // 改变线的连接节点
        this.jsPlumb.bind("connectionMoved", evt => {
          this.changeLine(evt.originalSourceId, evt.originalTargetId);
        });

        // 连线右击
        this.jsPlumb.bind("contextmenu", evt => {
          this.onContextmenu(evt);
          // console.log('contextmenu', evt)
          return false;
        });

        // 连线
        this.jsPlumb.bind("beforeDrop", evt => {
          let from = evt.sourceId;
          let to = evt.targetId;
          if (from === to) {
            this.$message.error("Nodes do not support connecting to themselves", 1000);
            return false;
          }
          if (this.hasLine(from, to)) {
            this.$message.error("This relationship already exists and duplicate creation is not allowed", 1000);
            return false;
          }
          // if (this.hashOppositeLine(from, to)) {
          //     this.$message.error('不支持两个节点之间连线回环',1000);
          //     return false
          // }
          this.$message.success("connection is successful", 1000);
          return true;
        });

        // beforeDetach
        this.jsPlumb.bind("beforeDetach", evt => {
          console.log("beforeDetach", evt);
        });
        this.jsPlumb.setContainer(this.$refs.efContainer);
      });
    },
    onContextmenu(event) {
      this.$contextmenu({
        items: [
          {
            label: "Delete from Diagram",
            icon: "el-icon-delete",
            onClick: () => {
              console.log("Delete from Diagram");
              this.deleteElement();
            }
          }
        ],
        event, // 鼠标事件信息
        customClass: "custom-class", // 自定义菜单 class
        zIndex: 3, // 菜单样式 z-index
        minWidth: 230 // 主菜单最小宽度
      });
      return false;
    },

    // 点击部署流图 
    dev_pipelline(){
      if (
        this.data.name == "undefined" ||
        this.data.name == undefined ||
        this.data.name == ""
      ) {
        this.$message({
          message: "Please select a diagram first",
          type: "warning",
          duration: 2000
        });
        return;
      }
      if (this.pipeLineType != "Deployment" || this.pipeLineType == "") {
        this.$message({
          message: "Please select the diagram category, and the category must be Deployment when deploying QA",
          type: "warning",
          duration: 3000
        });
        return;
      }
      // 检查是否有 Mutiplexer 组件 (推理必须有,并且必须以这个组件为开始)
      let muti_flag = false;
      let muti_count = 0;
      let outputBuilder_flag = false;
      let outputBuilder_count = 0;

      for (let i in this.data["nodeList"]) {
        if (this.data["nodeList"][i]["type"] == "Multiplexer") {
          muti_flag = true;
          muti_count += 1;
        }
        if (this.data["nodeList"][i]["type"] == "OutputBuilder") {
          outputBuilder_flag = true;
          outputBuilder_count += 1;
        }
      }
      if (muti_flag == false || muti_count > 1) {
        this.$message({
          message: "An inference diagram must and can only contain one Multiplexer component",
          type: "warning",
          duration: 3000
        });
        return;
      }
      if (outputBuilder_flag == false || outputBuilder_count > 1) {
        this.$message({
          message: "An inference diagram must and can only contain one OutputBuilder component",
          type: "warning",
          duration: 3000
        });
        return;
      }

      // 给 Multiplexer 组件设置初始值 "" 用于第一次运行
      for (let i in this.data["nodeList"]) {
        if (this.data["nodeList"][i]["type"] == "Multiplexer") {
          for (let j in this.data["nodeList"][i]["inputParams"]) {
            if (this.data["nodeList"][i]["inputParams"][j]["name"] == "value") {
              this.data["nodeList"][i]["inputParams"][j]["value"] = JSON.stringify({"question":""});
              break;
            }
          }
        }
      }
      console.log("🚀 -> this.data:\n", this.data);
      let tempJsonData = this.data;

      // 设置流图信息
      this.pipeline_components = [];
      this.pipeline_info = [];
      // 获取流图中的组件
      this.pipeline_components = tempJsonData['nodeList'].map(d => { return d});

      // 获取流图是否多次检索
      let retriever_flag = false;
      let controller_type = "";
      let thisMultiplexerId = "";
      for(let i in tempJsonData['nodeList']){
        if(tempJsonData['nodeList'][i]['type'].includes("Controller")){
          retriever_flag = true;
          controller_type = tempJsonData['nodeList'][i]['type'];
        }
        if(tempJsonData['nodeList'][i]['type'] == "Multiplexer"){
          thisMultiplexerId = tempJsonData['nodeList'][i]['id']
        }
      }

      for(let i in tempJsonData['lineList']){
        if(tempJsonData['lineList'][i]['to'] == thisMultiplexerId){
          retriever_flag = true;
          break;
        }
      }
      if(retriever_flag == true){
        if (controller_type != "" && controller_type.includes("Cok")){
          this.pipeline_info.push("Multiple Query")
        }else{
          this.pipeline_info.push("Multiple Retrieve")
        }
      }else{
        this.pipeline_info.push("Single  Retrieve")
      }

      // 判断知识库类型
      for(let i in tempJsonData['nodeList']){
          // 暂时通过controller 判断知识库类型
          if(tempJsonData['nodeList'][i]['type'].includes("Knowledge_") || tempJsonData['nodeList'][i]['type'].includes("KnowledgeCombiner") || tempJsonData['nodeList'][i]['type'].includes("KnowledgeSelector") ){

            for(let j in tempJsonData['nodeList'][i]['initParams']){
              if(tempJsonData['nodeList'][i]['initParams'][j]['name'] == 'tag'){
                this.pipeline_info.push(tempJsonData['nodeList'][i]['initParams'][j]['value'])
              }
            }
          }
      }

      // 更新pipline_data
      let message = this.$message({
        message: "Deploying, please wait",
        type: "success",
        duration: 0,
        showClose:false
      });
      this.data.pipeLineType = this.pipeLineType;
      // 清空加载的缓存
      utils.do_initCatch();
      utils.do_executePipeline(this,message,'dev');
    },


    // 加载流程图
    loadEasyFlow() {
      // 初始化节点
      for (var i = 0; i < this.data.nodeList.length; i++) {
        let node = this.data.nodeList[i];
        // 设置源点，可以拖出线连接其他节点
        this.jsPlumb.makeSource(
          node.id,
          lodash.merge(this.jsplumbSourceOptions, {})
        );
        // // 设置目标点，其他源点拖出的线可以连接该节点
        this.jsPlumb.makeTarget(node.id, this.jsplumbTargetOptions);
        if (!node.viewOnly) {
          this.jsPlumb.draggable(node.id, {
            containment: "parent",
            stop: function(el) {
              // 拖拽节点结束后的对调
              console.log("拖拽结束: ", el);
            }
          });
        }
      }
      // 初始化连线
      for (var i = 0; i < this.data.lineList.length; i++) {
        let line = this.data.lineList[i];
        var connParam = {
          source: line.from,
          target: line.to,
          label: line.label ? line.label : "",
          connector: line.connector ? line.connector : "",
          anchors: line.anchors ? line.anchors : undefined,
          paintStyle: line.paintStyle ? line.paintStyle : undefined
        };
        this.jsPlumb.connect(connParam, this.jsplumbConnectOptions);
      }
      this.$nextTick(function() {
        this.loadEasyFlowFinish = true;
      });
    },
    // 设置连线条件
    setLineLabel(from, to, label_from, label_to) {
      let label = label_from + " -> " + label_to;
      var conn = this.jsPlumb.getConnections({
        source: from,
        target: to
      })[0];
      if (!label || label === "") {
        conn.removeClass("flowLabel");
        conn.addClass("emptyFlowLabel");
      } else {
        conn.addClass("flowLabel");
      }
      conn.setLabel({
        label: label
      });
      this.data.lineList.forEach(function(line) {
        if (line.from == from && line.to == to) {
          line.label_from = label_from;
          line.label_to = label_to;
          line.label = label;
        }
      });
    },
    // 删除激活的元素
    deleteElement() {
      if (this.activeElement.type === "node") {
        this.deleteNode(this.activeElement.nodeId);
      } else if (this.activeElement.type === "line") {
        this.$confirm("Are you sure you want to delete the line you clicked on?", "prompt", {
          confirmButtonText: "Ok",
          cancelButtonText: "Return",
          type: "warning"
        })
          .then(() => {
            var conn = this.jsPlumb.getConnections({
              source: this.activeElement.sourceId,
              target: this.activeElement.targetId
            })[0];
            this.jsPlumb.deleteConnection(conn);
          })
          .catch(() => {});
      }
    },
    // 删除线
    deleteLine(from, to) {
      this.data.lineList = this.data.lineList.filter(function(line) {
        if (line.from == from && line.to == to) {
          return false;
        }
        return true;
      });
    },
    // 改变连线
    changeLine(oldFrom, oldTo) {
      this.deleteLine(oldFrom, oldTo);
    },
    // 改变节点的位置
    changeNodeSite(data) {
      for (var i = 0; i < this.data.nodeList.length; i++) {
        let node = this.data.nodeList[i];
        if (node.id === data.nodeId) {
          node.left = data.left;
          node.top = data.top;
        }
      }
    },
    /**
     * 拖拽结束后添加新的节点
     * @param evt
     * @param nodeMenu 被添加的节点对象
     * @param mousePosition 鼠标拖拽结束的坐标
     */
    addNode(evt, nodeMenu, mousePosition) {
      var screenX = evt.originalEvent.clientX,
        screenY = evt.originalEvent.clientY;
      let efContainer = this.$refs.efContainer;
      var containerRect = efContainer.getBoundingClientRect();
      var left = screenX,
        top = screenY;
      // 计算是否拖入到容器中
      if (
        left < containerRect.x ||
        left > containerRect.width + containerRect.x ||
        top < containerRect.y ||
        containerRect.y > containerRect.y + containerRect.height
      ) {
        this.$message.error("Please drag the node into the canvas", 1000);
        return;
      }
      left = left - containerRect.x + efContainer.scrollLeft;
      top = top - containerRect.y + efContainer.scrollTop;
      // 居中
      left -= 85;
      top -= 16;
      var nodeId = this.getUUID();
      // 动态生成名字
      var origName = nodeMenu.name;
      var nodeName = origName;
      var index = 1;
      while (index < 10000) {
        var repeat = false;
        for (var i = 0; i < this.data.nodeList.length; i++) {
          let node = this.data.nodeList[i];
          if (node.name === nodeName) {
            nodeName = origName + index;
            repeat = true;
          }
        }
        if (repeat) {
          index++;
          continue;
        }
        break;
      }
      var node = {
        id: nodeId,
        name: nodeName,
        type: nodeMenu.type,
        left: left + "px",
        top: top + "px",
        ico: nodeMenu.ico,
        state: "success",
        initParams: nodeMenu.initParams,
        inputParams: nodeMenu.inputParams,
        outputParams: nodeMenu.outputParams,
        components: nodeMenu.components,
        codeFilePath: nodeMenu.codeFilePath,
        description: nodeMenu.description,
      };

      /**
       * 这里可以进行业务判断、是否能够添加该节点
       */
      this.data.nodeList.push(node);
      this.$nextTick(function() {
        this.jsPlumb.makeSource(nodeId, this.jsplumbSourceOptions);
        this.jsPlumb.makeTarget(nodeId, this.jsplumbTargetOptions);
        this.jsPlumb.draggable(nodeId, {
          containment: "parent",
          stop: function(el) {
            // 拖拽节点结束后的对调
            console.log("拖拽结束: ", el);
          }
        });
      });
    },
    /**
     * 删除节点
     * @param nodeId 被删除节点的ID
     */
    deleteNode(nodeId) {
      this.$confirm(this.$t('topmenu.del_node') + nodeId + "?", "", {
        confirmButtonText: this.$t('common.ok'),
        cancelButtonText: this.$t('common.return'),
        type: "warning",
        closeOnClickModal: false
      })
        .then(() => {
          /**
           * 这里需要进行业务判断，是否可以删除
           */
          this.data.nodeList = this.data.nodeList.filter(function(node) {
            if (node.id === nodeId) {
              // 伪删除，将节点隐藏，否则会导致位置错位
              // node.show = false
              return false;
            }
            return true;
          });
          this.$nextTick(function() {
            this.jsPlumb.removeAllEndpoints(nodeId);
          });
        })
        .catch(() => {});
      return true;
    },
    clickNode(nodeId) {
      this.activeElement.type = "node";
      this.activeElement.nodeId = nodeId;
      this.$refs.nodeForm.nodeInit(this.data, nodeId);
    },
    // 是否具有该线
    hasLine(from, to) {
      for (var i = 0; i < this.data.lineList.length; i++) {
        var line = this.data.lineList[i];
        if (line.from === from && line.to === to) {
          return true;
        }
      }
      return false;
    },
    // 是否含有相反的线
    hashOppositeLine(from, to) {
      return this.hasLine(to, from);
    },
    nodeRightMenu(nodeId, evt) {
      this.menu.show = true;
      this.menu.curNodeId = nodeId;
      this.menu.left = evt.x + "px";
      this.menu.top = evt.y + "px";
    },
    repaintEverything() {
      this.jsPlumb.repaint();
    },

    getResultData() {
      utils.do_getResultData(this);
    },

    // 创建控制流图时 改变了选中值
    changeCommonCompent(value){
        console.log("🚀 -> value:\n", value)
        // console.log("🚀 -> initConfigDataJson:\n", initConfigDataJson)
        let tempConfig = []
        for(let i in initConfigDataJson){
          for(let j in initConfigDataJson[i]['children']){
            if(value.includes(initConfigDataJson[i]['children'][j]['type'])){
                // 使用深拷贝创建新对象
                tempConfig.push(JSON.parse(JSON.stringify(initConfigDataJson[i]['children'][j])));
            }
          }
        }
        this.controllComponent.tempControllerValueConfig = [].concat(tempConfig);
    },

    dataReload(data) {
      this.pageModel = "流图设计";
      this.QAModel = false;
      this.easyFlowVisible = false;
      this.data.name = data.name;
      this.data.nodeList = [];
      this.data.lineList = [];
      this.pipeLineType = data["pipeLineType"];
      this.QuestionData = "";
      this.AnswerData = "";
      this.RetrieverData = [];

      this.$nextTick(() => {
        data = lodash.cloneDeep(data);
        this.easyFlowVisible = true;
        this.data = data;
        this.$nextTick(() => {
          this.jsPlumb = jsPlumb.getInstance();
          this.$nextTick(() => {
            this.jsPlumbInit();
          });
        });

      });
    },

    // 获取所有pipeline的数据
    setPipelineData() {
      this.pipelineList = utils.do_getAllJsonData(this);
      this.pipeLineTypeOptions = this.pipelineList.map(item => {
        return item.value;
      });

      console.log("🚀 -> this.pipelineList :\n", this.pipelineList);
      console.log(
        "🚀 -> this.pipeLineTypeOptions:\n",
        this.pipeLineTypeOptions
      );
    },

    return_design_page(value){
      utils.do_initCatch();
      this.selectPipeline(value);
    },

    // 选择一个已有流程图
    selectPipeline(value) {
      if(value == undefined){
        value = this.thisPipeline
      }
      this.setPipelineData();
      let tempJsonData = null;
      for (let i in this.pipelineList) {
        if (this.pipelineList[i]["value"] === value[0]) {
          for (let j in this.pipelineList[i]["children"]) {
            if (this.pipelineList[i]["children"][j]["value"] === value[1]) {
              tempJsonData = this.pipelineList[i]["children"][j]["data"];
            }
          }
          this.pipeLineType = this.pipelineList[i].pipeLineType;
        }
      }
      console.log("🚀 -> tempJsonData:\n", tempJsonData);

      // reload 
      this.dataReload(tempJsonData)
     
    },

    // 动态检查是否符合linux文件夹命名规则,并替换非法字符
    handleInput() {
      this.data.name = this.data.name.replace(/[\/\0\*\<\>\|\:";\?\\ ]/g, '');
    },

    // 创建一个新流程图
    createPipe() {
      let temp_default_name = this.$t('topmenu.default_name');
      this.dataReload({ nodeList: [], lineList: [], name: temp_default_name });
      this.thisPipeline = "";
    },

    // 删除一个pipeline
    handleDeletePipeline(data) {
      console.log(data);
      this.$confirm("Confirm deletion?", "prompt", {
        confirmButtonText: "Ok",
        cancelButtonText: "Return",
        type: "warning"
      }).then(() => {
        utils.do_delPipelineByFileName(
          this,
          data["data"]["name"],
          data["data"]["pipeLineType"]
        );
      });
    },
    // 保存此流程图
    savePipeline() {
      console.log("this.data.name \n", this.data.name);
      if (
        this.data.name == "undefined" ||
        this.data.name == undefined ||
        this.data.name == ""
      ) {
        this.$message({
          message: "Please select a diagram first",
          type: "warning",
          duration: 1500
        });
        return;
      }
      console.log("this.pipeLineType \n", this.pipeLineType);
      if (this.pipeLineType == "" || this.pipeLineType == undefined) {
        this.$message({
          message: "Please select a diagram category",
          type: "warning",
          duration: 1500
        });
        return;
      }

      this.data.pipeLineType = this.pipeLineType;
      console.log("🚀 -> this.data:\n", this.data);
      utils.do_savePipeline(this);
    },

    // 执行此流程图
    runPipeline() {
      if (
        this.data.name == "undefined" ||
        this.data.name == undefined ||
        this.data.name == ""
      ) {
        this.$message({
          message: "Please select a diagram first",
          type: "warning",
          duration: 1500
        });
        return;
      }
      if (this.pipeLineType == "") {
        this.$message({
          message: "Please select a diagram category",
          type: "warning",
          duration: 1500
        });
        return;
      }
      // console.log("🚀 -> this.data:\n", this.data)
      this.$message({
        message: "Executing, please wait...",
        type: "success",
        duration: 3000
      });
      this.data.pipeLineType = this.pipeLineType;
      utils.do_initCatch();
      utils.do_executePipeline(this,null,null);
    },

    zoomAdd() {
      if (this.zoom >= 1) {
        return;
      }
      this.zoom = this.zoom + 0.1;
      this.$refs.efContainer.style.transform = `scale(${this.zoom})`;
      this.jsPlumb.setZoom(this.zoom);
    },
    zoomSub() {
      if (this.zoom <= 0) {
        return;
      }
      this.zoom = this.zoom - 0.1;
      this.$refs.efContainer.style.transform = `scale(${this.zoom})`;
      this.jsPlumb.setZoom(this.zoom);
    },
    // 下载数据
    downloadData() {
      this.$confirm("Are you sure you want to download this process data?", "prompt", {
        confirmButtonText: "Ok",
        cancelButtonText: "Return",
        type: "warning",
        closeOnClickModal: false
      })
        .then(() => {
          var datastr =
            "data:text/json;charset=utf-8," +
            encodeURIComponent(JSON.stringify(this.data, null, "\t"));
          var downloadAnchorNode = document.createElement("a");
          downloadAnchorNode.setAttribute("href", datastr);
          downloadAnchorNode.setAttribute("download", "data.json");
          downloadAnchorNode.click();
          downloadAnchorNode.remove();
          this.$message.success("Downloading, please wait...", 1500);
        })
        .catch(() => {});
    },
    openHelp() {
      this.flowHelpVisible = true;
      this.$nextTick(function() {
        this.$refs.flowHelp.init();
      });
    },

    // 创建基础组件
    createCommonElement() {
      if (this.newComponent.name == "") {
        this.$message({
          message: "The base component type or name cannot be empty",
          type: "warning",
          duration: 2000
        });
        return;
      }
      let codeFilePath = document.getElementById("newComponent_filePath").innerText;
      codeFilePath = codeFilePath.replace(/\s/g, '');
      console.log("🚀 -> codeFilePath:\n", codeFilePath)

      if (codeFilePath == "") {
        this.$message({
          message: "The base component path cannot be empty",
          type: "warning",
          duration: 2000
        });
        return;
      }
      let saveNode = {
        parent_type: this.config_parent_list[this.newComponent.parent_type].type,
        id: new Date().toISOString() + "_component",
        name: this.newComponent.name,
        ico: "el-icon-time",
        type: this.newComponent.name,
        initParams: [],
        inputParams: [],
        outputParams: [],
        components: [],
        codeFilePath: codeFilePath,
        description: this.newComponent.description,
      }
      console.log("🚀 -> saveNode:\n", saveNode)

      // //保存
      utils.do_saveConfigData(this, initConfigDataJson, "save", saveNode);
      // console.log("创建了基础组件");
      this.dialogVisible_createElement = false;
      // // 刷新
      window.location.reload();
    },

    // 创建控制流图
    createControllElement() {
      console.log("🚀 -> this.controllComponent:\n", this.controllComponent)

      if (this.controllComponent.children_componentList.length == 0) {
        this.$message({
          message: "Please add subcomponents",
          type: "warning",
          duration: 2000
        });
        return;
      }

      let codeFilePath = document.getElementById("controllerComponent_filePath").innerText;
      codeFilePath = codeFilePath.replace(/\s/g, '');
      console.log("🚀 -> codeFilePath:\n", codeFilePath)

      if (this.controllComponent.name == "") {
        this.$message({
          message: "Please set the component name",
          type: "warning",
          duration: 2000
        });
        return;
      }
      if (codeFilePath == "") {
        this.$message({
          message: "Control component path cannot be empty",
          type: "warning",
          duration: 2000
        });
        return;
      }
      var temp_initParamsJson = null;
      var temp_inputParamsJson = null;
      var temp_outputParamsJson = null;
      try {
        temp_initParamsJson = JSON.parse(this.controllComponent.initParams);
        temp_inputParamsJson = JSON.parse(this.controllComponent.inputParams);
        temp_outputParamsJson = JSON.parse(this.controllComponent.outputParams);
      } catch (error) {
        this.$message({
          message: "Parameter list serialization failed. Please check",
          type: "warning",
          duration: 2000
        });
        return;
      }

      let saveNode = {
        id: new Date().toISOString() + "_controlcomponent",
        name: this.controllComponent.name,
        ico: "el-icon-time",
        type: this.controllComponent.name,
        initParams: temp_initParamsJson,
        tempControllerValueConfig:this.controllComponent.tempControllerValueConfig,
        inputParams: temp_inputParamsJson,
        outputParams: temp_outputParamsJson,
        components: this.controllComponent.children_componentList,
        codeFilePath:codeFilePath,
        description: this.controllComponent.description,
      };
      console.log("🚀 -> initConfigDataJson:\n", initConfigDataJson);
      //保存
      utils.do_saveConfigData(this, initConfigDataJson, "save", saveNode);
      this.dialogVisible_createControllElement = false;
      // 刷新
      window.location.reload();
    },
      

    isJson(obj) {
      return typeof obj === "object" && obj !== null && !Array.isArray(obj);
    }
  }
};
</script>

<style>

  .ef-tooltar {

    display: flex;
    justify-content: space-around;
    align-items: center;
    
    padding-left: 10px;
    box-sizing: border-box;
    /* line-height: 42px; */
    z-index: 3;
    border-bottom:none;
  }

  .title-logo {
    height: 80px;
    width: 170px;
    float: left;
    background-image: url('../../assets/images/logo.png');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
  }

  .right-model {
    color: #BA5511;
    font-size: 1.2rem;

  }

  .el-link.el-link--default {
      padding: 7px;
      color: #BA5511;
      font-size:1.1rem;
      border-radius: 15px;
      border: solid 1px transparent;
      background-color: transparent;
      /* box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5); */
  }


  
  .title-edit {
      margin-left: 10px;
  }

  .title-right{
      margin-left: 10px;
      float: right;
      margin-right: 5px
  }


    /* 设置滚动条的样式 */
    ::-webkit-scrollbar {
        width: 5px; /* 设置滚动条宽度 */
        padding: 0px;
        border: 1px solid transparent !important; /* 添加底部边框并设置为透明 */
    }

    /* 设置滚动条轨道的样式 */
    ::-webkit-scrollbar-track {
        background-color: transparent; /* 设置滚动条轨道背景颜色 */
        padding: 0px;
        border: 1px solid transparent !important; /* 添加底部边框并设置为透明 */
    }

    ::-webkit-scrollbar-corner {
        background-color: transparent !important; /* 设置滚动条角落背景颜色 */
    }
    /* 设置滚动条滑块的样式 */
    ::-webkit-scrollbar-thumb {
        background-color: #1E2227; /* 设置滚动条滑块颜色 */
    }
    .el-upload-dragger {
        background-color: none;
        border: solid 2px #23272E;
    }


    .el-divider {
      background-color: #21AA93;

    }

    /* 左上角 pipeline name */
    .el-link.el-link--primary {
        color: #21AA93;
        font-size: 1.05rem;
    }

    /* 文字按钮 */
    .el-button--text{
      color:#21AA93;
    }

    /* 单选下拉框 */
    .el-select .el-input,
    .el-select .el-input input {
      color: #906E33;
      background-color: transparent;
      /* border : solid 1px #1E2227; */
      
    }
    .el-select-dropdown {
        border: 2px solid #1E2227;
        border-radius: 4px;
        background-color: #303133;
    }

    .el-select-dropdown__item {
      color: #906E33;
    }

    .el-select .el-tag {
      border: solid 2px #1E2227 !important;
      background-color: transparent !important;
      box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.5);
  }

    .el-select-dropdown__item.hover, .el-select-dropdown__item:hover {
        background-color: #1E2227;
    }

    .el-select-dropdown.is-multiple .el-select-dropdown__item.selected {
        color: #409EFF;
        background-color: transparent;
    }

    /* 级联下拉框 */
    .el-cascader-menu {
        background-color: transparent;
        border:null;
    }

    .el-cascader-menu-item {
        color: #21AA93;
        border:null;
    }
    .el-cascader-panel {
        border: 2px solid #1E2227;
        border-radius: 4px;
        background-color: #303133;
    }

    .el-cascader-node__label {
      color: #906E33;
    }

    .el-cascader-node:not(.is-disabled):hover {
      background-color: #1E2227;
    }
    .el-cascader-node:not(.is-disabled):focus, .el-cascader-node:not(.is-disabled):hover {
        background: #1E2227;
    }

    .ef-node-form-header {
        height: 32px;
        color: #21AA93;
        line-height: 32px;
        padding-left: 12px;
        font-size: 1.05rem;
    }


    .el-input__inner {
      background-color: transparent;
      color: #906E33;
      border : solid 2px #1E2227;
      box-shadow: inset -1px 1px 3px #1E2227,
                  inset 1px -1px 3px #1E2227;
    }

    .el-input.is-disabled .el-input__inner {
      background-color: #23272E;
      color: #906E33;
      border : solid 2px #1E2227;
      box-shadow: inset -1px 1px 3px #1E2227,
                  inset 1px -1px 3px #1E2227;
    }

    .el-input-group__prepend {
      background-color: transparent;

    }

    .el-input-group__append {
      background-color: transparent;
    }


    .el-textarea__inner {
      background-color: transparent;
      color: #906E33;
      border : solid 2px #1E2227;
      box-shadow: inset -1px 1px 3px #1E2227,
                  inset 1px -1px 3px #1E2227;
    }

    .textarea-param {

        color:#88878a;
        border-radius:4px;
        padding:5px;
        /* width:calc(100% - 20px); */
        height:100px;
        background-color: transparent;
        border : solid 2px #1E2227;
        box-shadow: inset -1px 1px 3px #1E2227,
                    inset 1px -1px 3px #1E2227;


    }
    textarea:disabled {
      width:calc(100% - 15px);
      background-color: #23272E;
      color: #906E33;
      border : solid 2px #1E2227;
      box-shadow: inset -1px 1px 3px #1E2227,
                  inset 1px -1px 3px #1E2227;
    }

    .el-tag {
      color: #19BA8F;
      border: solid 3px transparent;
      background-color: transparent;
      box-shadow: 0px 0px 4px rgba(0, 0, 0, 0.5);

    }

    

    .ef-node-form {
      height: 100%;
    }

    .ef-node-form-header {
      height: 32px;
      background: #F1F3F4;
      color: #21AA93;
      line-height: 32px;
      padding-left: 12px;
      font-size: 1.05rem;
      background-color: transparent;
      border-top:none;
      border-bottom: solid 3px #303133;
      box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.5);
    }
    .ef-node-form-body {
      overflow-y:auto;
      height: calc(100% - 85px);
      padding-right:15px;
    }

    .el-input-group__append, .el-input-group__prepend {
        /* background-color: #F5F7FA; */
        color: #909399;
        border: 1px solid transparent;
        border-radius: 4px;
    }


.root {
  height: calc(100vh); 
  background-color: #303133;
}

.page{
  display: flex; 
  height: calc(100% - 125px);
  margin-top:30px;
}


.CodeMirror {
  position: relative;
  /* height: 100vh; */
  overflow: hidden;
  margin-top: 10px;
}
.cm-s-base16-light span.cm-string {
  color: #1f5c8b;
}

.edit {
  width: calc(100% - 100px);
}

.el-row {
  box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.5); /* 添加中间阴影效果 */
}

.left_div {
  overflow-y: auto;
  box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.5); /* 添加中间阴影效果 */
  border : solid 1px #1E2227;
  border-right:none;
  border-radius:20px;
  margin-left:15px;
}

.container {
  margin-left:20px;
  box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.5); 
  border : solid 1px #1E2227;
  border-radius:20px;
}

.right_div {
  width: 350px;
  /* overflow-y: auto; */
  margin-left:20px;
  box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.5); /* 添加中间阴影效果 */
  border : solid 1px #1E2227;
  border-radius:20px;
  margin-right:15px;
}

.ef-node-pmenu {
  color: #ba5511
}
.ef-node-pmenu:hover {
    color: #ba5511;
    background-color: transparent;
}

.ef-node-menu-li {
  color: #00ba8f ;
  box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5); /* 添加中间阴影效果 */
  border : solid 1px #1E2227;
  border-radius:20px;
  word-wrap: break-word;
  width:170px;
  padding : 10px 10px 10px 10px;

}
.ef-node-menu-li:hover {
  background-color: #1E2227 ;
  color: #00ba8f ;
  box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5); /* 添加中间阴影效果 */
  border : solid 1px #1E2227;
  border-radius:20px;
}

.ef-node-text {
  color: #00BA8F;
}

.ef-node-container {
  color: #FEE698 ;
  background-color : transparent;
  border : solid 1px transparent;
  box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5); /* 添加中间阴影效果 */

}

.ef-node-container:hover {
  color: #FEE698 ;
  background-color : transparent;
  border : solid 1px transparent;
  box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5); /* 添加中间阴影效果 */
}

.item {
  color: #3d6dba ;
  background-color: transparent;
  border : solid 1px transparent;
  box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5); /* 添加中间阴影效果 */

}

.jtk-overlay.flowLabel:not(.aLabel) {
    color: #906E33  !important;
    border: solid 2px  #23272E;
    background-color: #23272E !important;;
}

.my-el-button {
    color:#21AA93; 
    border: solid 1px transparent;
    background-color: transparent;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
}
.my-el-button:hover {
    color:#21AA93; 
    border: solid 1px transparent;
    background-color: #252527;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
}

.my-del_el-button {
    color: #F56C6C; 
    border: none;
    background-color: transparent;
}
.my-del_el-button:hover {
    color: #F56C6C; 
    border: none;
    background-color: #252527;
}

/* dialog */
.el-dialog {
  background-color : #303133;
}
.el-dialog__title {
  color: #00ba8f;
  font-size :1.20rem;
}


li {
  user-select: none;
}

/* screenshot */
/* #efContainer{
  background-color: white;
}
.ef-node-container{
  border: solid 2px #1E2227;
  color:black;
  -webkit-box-shadow: none !important;
  box-shadow: none !important;
}
.ef-node-text{
  color:black;
}
.ef-node-right-ico{
  color:black !important;
}
.nodeDescription{
  color:black !important;
}
.jtk-overlay.flowLabel:not(.aLabel) {
    color: black  !important;
    border: solid 2px  #23272E;
    background-color: white !important;
}
.item{
  color:black;
  border: solid 1px #1E2227;
  -webkit-box-shadow: none !important;
  box-shadow: none !important;
}
.arrow {
    width: 0;
    height: 0;
    border-left: 20px solid transparent;
    border-right: 20px solid transparent;
    border-bottom: 20px solid black;
}
.ef-node-left{
  color:black !important;
  background-color: black !important;
}
.ef-node-right{
  color:black !important;
  background-color: black !important;
}
.el-node-state-success{
  color:black;
}
path {
  color: black  !important;
  background-color: black  !important;
  stroke: black;
  stroke-width: 1.1;
  fill:none;
  pointer-events:visibleStroke;
}

.el-dialog {
    background-color: white;
}

.el-form {
  background-color: white;
}

.el-input.is-disabled .el-input__inner {
  background-color: white;
}

.el-input__inner {
  background-color: white;
}
.textarea{
  background-color: white !important;
}
textarea:disabled {
  background-color: white !important;
} */

</style>


