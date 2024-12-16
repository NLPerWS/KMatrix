<template>
    <div class="ef-node-form">
        <div class="ef-node-form-header">
            {{ $t("nodemenu.edit") }} 
        </div>
        <div class="ef-node-form-body" >
            <el-form :model="node" ref="dataForm" label-width="80px" v-show="type === 'node'">
                <el-form-item :label="$t('nodemenu.type')">
                    <el-input v-model="node.type" :disabled="true"></el-input>
                </el-form-item>
                <el-form-item :label="$t('nodemenu.name')">
                    <el-input v-model="node.name"></el-input>
                </el-form-item>

                <el-form-item :label="$t('nodemenu.description')">
                    <el-input type="textarea" rows="5" v-model="node.description"></el-input>
                </el-form-item>

                <el-form-item :label="$t('nodemenu.init_params')">

                    <div  v-for="(tempObj, index) in node.initParams" :key="index" >

                        <div v-if="tempObj.type == 'upload' ">

                            <el-button  size="small" class="my-el-button" @click="dialogVisible_upload=true;" >
                                {{node.initParams[0].name}}
                            </el-button>

                            <el-dialog 
                                :append-to-body=true
                                :visible.sync="dialogVisible_upload"
                                width="400px">
                                    <el-upload ref="upload"
                                        drag
                                        class="upload-demo"
                                        action=""
                                        multiple
                                        :accept=node.initParams[0].value
                                        :file-list="fileList"
                                        :http-request="myUploadFile"
                                        >
                                        <i class="el-icon-upload"></i>
                                        <div class="el-upload__text">Drag the file here or <em>click Upload</em></div>
                                    </el-upload>
                                <span slot="footer" class="dialog-footer">
                                    <el-button class="my-el-button" @click="dialogVisible_upload = false">{{ $t('common.return') }}</el-button>
                                    <el-button class="my-el-button" @click="submitUpload()">{{ $t('common.ok') }}</el-button>
                                </span>
                            </el-dialog>

                        </div>


                        <div v-else>
                            <el-tag size="medium">{{ $t('nodemenu.param_name') }}</el-tag>
                            <textarea  v-text="tempObj.name" :disabled="true" class="textarea-param"></textarea>

                            <el-tag size="medium">{{ $t('nodemenu.param_type') }}</el-tag>
                            <textarea  v-text="tempObj.type" :disabled="true" class="textarea-param"></textarea>

                            <el-tag size="medium">{{ $t('nodemenu.param_value') }}</el-tag>
                            <el-input type="textarea" rows="5"  style="margin-bottom: 3px;" v-model="node.initParams[index].value" />

                        </div>

                        <el-divider></el-divider>

                    </div>

                </el-form-item>

                <el-form-item label="">
                    <el-button round class="my-el-button"  @click="look_running_data">{{ $t('nodemenu.runtime_value') }}</el-button>
                </el-form-item>

                <el-dialog
                    :title="$t('nodemenu.runtime_value')"
                    :visible.sync="dialogVisible_look_variable"
                    width="80%">
                    <codemirror
                            :value="variable_data"
                            class="code"
                    ></codemirror>

                </el-dialog>

                <el-form-item>
                    <el-button class="my-el-button" round icon="el-icon-check" @click="save">{{ $t('nodemenu.save') }}</el-button>
                </el-form-item>
            </el-form>

            <el-form :model="line" ref="dataForm" label-width="80px" v-show="type === 'line'">

                <el-form-item :label="$t('nodemenu.label_from')">
                    <!-- <el-input v-model="line.label_from"></el-input> -->
                    <el-select v-model="line.label_from" placeholder="please select label_from">
                        <el-option v-for="item in line.label_from_outputList" :key="item" :label="item"
                            :value="item">
                        </el-option>
                    </el-select>


                </el-form-item>
                <el-form-item :label="$t('nodemenu.label_to')">
                    <el-select v-model="line.label_to" placeholder="please select label_from">
                        <el-option v-for="item in line.label_to_inputList" :key="item" :label="item"
                            :value="item">
                        </el-option>
                    </el-select>
                </el-form-item>


                <el-form-item>
                    <el-button round class="my-el-button" icon="el-icon-check" @click="saveLine">{{ $t('nodemenu.save') }}</el-button>
                </el-form-item>
            </el-form>
        </div>
    </div>

</template>

<script>
import { cloneDeep } from 'lodash'
import * as utils from "@/assets/js/utils";
import { rootUrl } from "@/assets/js/host";


export default {
    props: {
        pipelineName: {
            default:null,
            required: true
        },
        pipelineType: {
            default:null,
            required: true
        }
    },

    data() {
        return {
            dialogVisible_upload:false,
            dialogVisible_look_variable:false,
            variable_data:"",
            visible: true,
            // node 或 line
            type: 'node',
            node: {},
            line: {},
            data: {},
            stateList: [{
                state: 'success',
                label: 'success'
            }, {
                state: 'warning',
                label: 'warning'
            }, {
                state: 'error',
                label: 'error'
            }, {
                state: 'running',
                label: 'running'
            }],

            fileList:[],
            formData:new FormData(),
        }
    },
    methods: {
        /**
         * 表单修改，这里可以根据传入的ID进行业务信息获取
         * @param data
         * @param id
         */
        nodeInit(data, id) {
            this.type = 'node'
            this.data = data
            data.nodeList.filter((node) => {
                if (node.id === id) {
                    this.node = cloneDeep(node)
                }
            })
        },
        lineInit(line) {
            this.type = 'line'
            this.line = line

        },
        
        // 查看组件运行时变量
        look_running_data(){
            console.log("🚀 -> this:\n", this.data )
            let result = utils.do_getVariableData(this,this.pipelineName,this.pipelineType);
            // // 数组对象格式化的逻辑 JSON.stringify的第三个属性就是让我们格式化代码用的，直接传入数字x（10以内），就表示前面是x个空格的距离，我用的是2，也可以用'\t'，这样就是一个tab的距离了。
            this.variable_data = JSON.stringify(result, null ,2) 
            this.dialogVisible_look_variable = true;
        },


        // 修改连线
        saveLine() {
            this.$emit('setLineLabel', this.line.from, this.line.to, this.line.label_from, this.line.label_to)
        },
        save() {
            this.data.nodeList.filter((node) => {
                if (node.id === this.node.id) {
                    node.name = this.node.name
                    node.left = this.node.left
                    node.top = this.node.top
                    node.ico = this.node.ico
                    node.state = this.node.state
                    node.initParams = this.node.initParams
                    node.inputParams = this.node.inputParams
                    node.outputParams = this.node.outputParams
                    node.components = this.node.components
                    node.codeFilePath = this.node.codeFilePath
                    node.description = this.node.description
                    node.tag = this.node.tag
                    this.$emit('repaintEverything')
                }
            })
        },

        // 多文件上传
        // 覆盖默认上传行为
        myUploadFile(file){
            this.formData.append("files",file.file);
        },
        // 确定提交
        submitUpload(){

            let savePath = "";
            for(let i in this.node.initParams){
                if(this.node.initParams[i].name == "savePath"){
                    savePath = this.node.initParams[i].value;
                    break;
                }
            }
            if (savePath == ""){
                this.$message({
                    message: "Please fill in the save path first",
                    type: "error",
                    duration:1500
                });
                return;
            }

            this.formData.append("savePath",savePath);
            let that = this;
            $.ajax({
                type:"post",
                catch:false,
                dataType:"json",
                processData: false,
                contentType:false,
                async: false,
                url:rootUrl+"/uploadKnowledge",
                data:this.formData,
                success:function(data){
                    if(data.code == 200){
                        that.$message({
                            message: data.data,
                            type: "success",
                            duration:1500
                        });
                        that.fileList = [];
                        that.formData = new FormData();
                    }else{
                        that.$message({
                            message: data.data,
                            type: "error",
                            duration:1500
                        });
                        return;
                    }
                },
                error:function(err){
                    that.$message({
                            message: "Network exception, please try again later",
                            type: "error",
                            duration:1500
                        });
                    return;
                }
            });
            this.dialogVisible_upload = false;
        }
    }
}
</script>

<style scope>
.el-node-form-tag {
    position: absolute;
    top: 50%;
    margin-left: -15px;
    height: 40px;
    width: 15px;
    background-color: #fbfbfb;
    border: 1px solid rgb(220, 227, 232);
    border-right: none;
    z-index: 0;
}




</style>
