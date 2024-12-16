<template>

    <div id="layout" class="theme-cyan" style="width: 100%; display: flex; height: 100%;">
      
        <div class="left_div  py-xl-4 py-3 px-xl-4 px-3" style="border-radius: 20px;  width: 350px; overflow-y: auto;" >
            
              <div style="height: 100%;"
                  class="tab-pane fade show active"
                  id="nav-tab-chat"
                  role="tabpanel">

                  <!-- <div class="logo">


                  </div> -->


                  <div class="d-flex  align-items-center mb-4">

                    <div class="avatar rounded-circle no-image bg-primary text-light">
                        <span><i class=" el-icon-date"></i></span>
                    </div>

                      <h4 style="margin-left: 20px;" class="mb-0 text-primary ">{{ $t('qa.pipeline_xq_title') }}</h4>
                  </div>

                  <div class="form-group input-group-lg search mb-3" style="height: calc(100% - 50px); display: flex; flex-direction: column;overflow-y: auto;">
                      <div class="pipelineinfo">
                        <h6 class="pipelineinfo-title  text-primary "><strong>{{ $t('qa.pipeline_coms') }}</strong></h6>

                        <ul>
                          <li v-for="(item, index) in pipeline_components" :key="index">
                            {{ item.name }}
                            <ul v-if="item.components && item.components.length > 0">
                              <li v-for="(subItem, subIndex) in item.components" :key="subIndex">
                                <span style="white-space: break-spaces;" v-text="subItem"></span>
                              </li>
                            </ul>
                          </li>
                        </ul>

                      </div>

                      <div class="pipelineinfo">
                        <h6 class="pipelineinfo-title text-primary"><strong>{{ $t('qa.pipeline_info') }}</strong></h6>
                        <ul>
                          <li v-for="item in pipeline_info" :key="item">
                            <span style="white-space: break-spaces;" v-text="item"></span>
                          </li>
                        </ul>
                      </div>

                  </div>

              </div>

        </div>

        <!-- <el-divider direction="vertical"></el-divider> -->

        <div class="inner_div main px-xl-5 px-lg-4 px-3">
          <div class="chat-body" style="height: 100%;">
            <div class="chat-header py-xl-4 py-md-3 py-2">
              <div class="container-xxl">
                <div class="row align-items-center">
                  <div class="">
                    <div class="media">
                      <div
                        class="avatar me-3 show-user-detail"
                        data-toggle="tooltip"
                        title=""
                        data-original-title=""
                      >
                        <div
                          class="avatar rounded-circle no-image bg-primary text-light"
                        >
                          <span><i class="el-icon-message-solid"></i></span>
                        </div>
                      </div>
                      <div class="media-body overflow-hidden">
                        <div class="d-flex align-items-center mb-1">
                          <h3 class="mb-0 text-primary">{{ $t('qa.pipeline_infer') }}</h3>
                        </div>
                        <div class="text-truncate"></div>
                      </div>
                    </div>
                  </div>

                </div>
              </div>
            </div>

            <div v-text="QuestionData" class="input_show"></div>



            <div class="output_show" v-text="AnswerData">
            </div>



            
            <div class="input-group align-items-center" style="display: flex;flex-direction: row;flex-wrap: nowrap;">

              <el-select style="width: 100%;" popper-class="popper-class" 
                @visible-change="click_input"
                v-model="QuestionData"
                clearable 
                filterable
                allow-create
                default-first-option
                :placeholder="$t('qa.input')">
                <el-option
                  v-for="item in QuestionDataList"
                  :key="item"
                  :label="item"
                  :value="item">

                </el-option>
              </el-select>

              <div class="input-group-append">
                <span class="input-group-text border-0 pr-0">

                  <el-button type="primary" class="btn btn-primary" @click="do_QaPage" round :loading="infer_loading">{{ $t('qa.send') }}&nbsp;<i class="el-icon-arrow-right"></i></el-button>

                </span>
              </div>

            </div>


          </div>
        </div>

        <!-- <el-divider direction="vertical" ></el-divider> -->
        <div class="right_div border-end py-xl-4 py-3 px-xl-4 px-3" style="border-radius: 20px; width: 350px;">
            <div class="tab-content ">
                
                <div
                    class="tab-pane fade show active"
                    id="nav-tab-chat"
                    role="tabpanel">


                    <div class="d-flex align-items-center mb-4">
                        <div class="avatar rounded-circle no-image bg-primary text-light">
                          <span><i class=" el-icon-tickets"></i></span>
                        </div>
                        <h4 style="margin-left: 20px;" class="mb-0 text-primary">{{ $t('qa.log') }}</h4>
                    </div>

                    <div class="form-group input-group-lg search mb-3">
                        
                        <div class="logDiv" :style="{ height: tableHeight }">
                          <div  v-for="(item, index) in RetrieverData" :key="index" >
                            <p class="log-line-text" v-html="item.text"></p>
                            <p v-if="index >= 2 && (RetrieverData[index].text.includes('search returned')  || RetrieverData[index].text.includes('查询返回') || RetrieverData[index].text.includes('Returns generator reply'))" class="el-icon-bottom log-line-arrow"></p>
                          </div>

                          <div v-if="RetrieverData.length != 0 " class="log-line-text" >done !!! </div>
                        </div>
                        


                    </div>

                </div>

            </div>
        </div>


    </div>

</template>

  <style  scoped>
    @import "../../assets/hoppinzq/chat/static/css/style.min.css";


    /* logo 像素 为 300 x 120 */
    .logo {
      height: 150px;
      background-image: url('../../assets/images/logo.png');
      background-size: contain;
      background-repeat: no-repeat;
      background-position: center;
    }

    .left_div {
      margin-left:10px;
      margin-right:5px;
      background-color: #303133;
      border: solid 1px #1E2227;
      box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.5); /* 添加中间阴影效果 */
    }

    .inner_div{
      margin-left:5px;

      background-color: #303133;
      border: solid 1px #1E2227;
      border-radius:20px;
      box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.5); /* 添加中间阴影效果 */
    }

    .right_div{
      margin-left:10px;
      margin-right:10px;
      background-color: #303133;
      border: solid 1px #1E2227;
      box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.5); /* 添加中间阴影效果 */
    }

    .avatar {
          display: flex;
          justify-content: center;
          align-items: center;
      }

    .main{
        height: 100%;
        order:initial;
        margin-right: 0px;
        transition: all .2s ease-in-out;
        overflow: hidden;
        -webkit-box-flex: 1;
        -ms-flex: 1;
        flex: 1;
        -webkit-box-ordinal-group: initial;
        -ms-flex-order: initial;
    }
    .chat-body{
      height: 100%;
    }

    li{
      word-wrap: break-word;
    }
    .el-divider--vertical{
      height: 100%;
    }
    .border-bottom {
        /* border-bottom: 1px solid #dee2e6 !important; */
    }
    .border-end {
        /* border-right: 1px solid #dee2e6 !important; */
    }
    .pipelineinfo-title{
      margin-left: 30px;
    }

    .pipelineinfo{
      flex: 1;
      /* height: 200px; */
      overflow-y: auto;
      border-radius: 10px;
      font-size: 1.02rem;
      padding-top: 10px;
      padding-bottom: 10px;
      padding-right: 10px;
      margin-bottom: 20px;
      background-color: #23272E;
      color: #9B9CA0;
    }


    .input_show{
      white-space: pre-wrap;
      word-wrap: break-word;
      border-radius: 10px;
      background-color: #0F1117;
      color: #626569;
      padding: 15px 15px 15px 15px;
      font-size: 1.02rem;
      overflow-y: auto;
      height: 100px;
      max-height: 100px;
    }

    .output_show{
      margin-top:30px;
      margin-bottom: 30px;
      padding: 15px 15px 15px 15px;

      color: #78828a ;
      border-radius: 10px;
      white-space: pre-wrap;
      word-wrap: break-word;
      overflow: auto;
      height: calc(100% - 360px);
      background-color: #0F1117;
      font-size: 1.02rem;
    }

    .logDiv{
      overflow-y:auto;
      color: #9B967D;
      word-wrap: break-word;
    }

    .el-icon-bottom {
      display: block;
      margin: 0 auto;
      text-align: center;
    }

    .log-line-text {
      margin-left: 2px;
      margin-top: 5px;
      margin-bottom: 5px;
      margin-right: 5px;

      padding: 5px;
      box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
      border-radius: 10px;
      border: solid 1px transparent;
      white-space: break-spaces;
      letter-spacing: 0.1rem;
      line-height: 1.5;

    }

    .log-line-arrow {
      margin-top: 20px;
      margin-bottom: 20px;

    }

    .el-input >>> .el-input__inner{
      background-color: #0F1117;
      border: solid 0px #0F1117;
    }
    .popper-class .el-select-dropdown__item {
      width: 300px;
      display: inline-block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

  </style>




<script>


import { ref, onMounted } from "vue";
import * as utils from "@/assets/js/utils";


export default {
  props: {
    data: Object,
    pipeLineType: String,
    pipeline_components:Array,
    pipeline_info:Array,
  },
  data() {
    return {
      showheader:false,
      QuestionData: "",
      QuestionDataList:[

      ],
      AnswerData: "",
      RetrieverData: [],
      infer_loading: false,
      tableHeight :  document.getElementById('app').clientHeight - 235 + "px",
    };
  },
  methods: {

    mounted() {
      console.log("QA");

    },

    click_input(flag){
      console.log(flag);
      if (flag == false){
        return
      }
      this.QuestionDataList = [
      ]

    },

    tableRowClassName({ row, rowIndex }) {
      console.log(rowIndex);
      if (rowIndex % 2 === 0) {
        return 'one-row';
      } else {
        return 'two-row';
      }
    },

    strIsJson(str) {
      if (typeof str !== 'string') {
        return false;
      }
      try {
        const json = JSON.parse(str);
        return typeof json === 'object';
      } catch (e) {
        return false;
      }
    },

    // 执行QA页面
    do_QaPage() {

      // 给 Multiplexer 组件设置初始值
      for (let i in this.data["nodeList"]) {
        if (this.data["nodeList"][i]["type"] == "Multiplexer") {
          for (let j in this.data["nodeList"][i]["inputParams"]) {
            if (this.data["nodeList"][i]["inputParams"][j]["name"] == "value") {
              // 判断 如果输入的是序列化格式 就直接使用,否则就序列化为含有input的json
                if (this.strIsJson(this.QuestionData)){
                  this.data["nodeList"][i]["inputParams"][j]["value"] = this.QuestionData;
                  console.log( "QA 数据为 json...\n", this.data["nodeList"][i]["inputParams"][j]["value"]);
                }else{
                  this.data["nodeList"][i]["inputParams"][j]["value"] = JSON.stringify({ "question": String(this.QuestionData) });
                  console.log("QA 数据不为 json...\n",this.data["nodeList"][i]["inputParams"][j]["value"]);
                }
              break;
            }
          }
        }
      }
      console.log("🚀 -> this.data:\n", this.data);
      this.$message({
        message: "Executing, please wait....",
        type: "success",
        duration: 3000
      });
      this.data.pipeLineType = this.pipeLineType;
      this.infer_loading = true;
      this.AnswerData = "";
      this.RetrieverData = []
      utils.do_executePipeline_QA(this);

    },

  }
};

</script>
