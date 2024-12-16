<template>
    <el-dialog
            title="Help"
            :visible.sync="dialogVisible"
            width="70%"
            customClass="flowHelp"
    >
        <el-tabs tab-position="left">
            <el-tab-pane label="如何修改流程图名称">
                <el-divider content-position="left">如何修改流程图名称</el-divider>
                <div class="info">点击左上角的文字(当前流程图名称)即可</div>
                <div class="info">注意: 如果名称与已有名称重复,则会覆盖已有流程图。</div>
            </el-tab-pane>


            <el-tab-pane label="如何新增普通组件">
                <el-divider content-position="left">如何新增普通组件</el-divider>
                <div class="info">点击左上角的 <p class="el-icon-edit"></p> 图标即可</div>
            </el-tab-pane>
            <el-tab-pane label="如何新增控制组件">
                <el-divider content-position="left">如何新增控制组件</el-divider>
                <div class="info">点击左上角的 <p class="el-icon-cpu"></p> 图标即可</div>
            </el-tab-pane>
            <el-tab-pane label="如何添加节点">
                <el-divider content-position="left">如何新增向流程图中添加组件</el-divider>
                <div class="info">按住鼠标拖拽左侧组件到中间画布中松开鼠标即可</div>
            </el-tab-pane>
            <el-tab-pane label="如何删除节点">
                <el-divider content-position="left">如何删除流程图中的组件</el-divider>
                <div class="info">
                    鼠标点中需要删除的节点，点击左上角的 <p class="el-icon-delete"></p> 图标
                </div>
            </el-tab-pane>
            <el-tab-pane label="如何移动节点">
                <el-divider content-position="left">如何移动流程图中的组件</el-divider>
                <div class="info">鼠标移动到节点中，当鼠标变为可拖拽的图标时按下鼠标移动到新的位置松开鼠标</div>
            </el-tab-pane>
            <el-tab-pane label="如何连线">
                <el-divider content-position="left">如何将流程图进行连线</el-divider>
                <div class="info">鼠标移动到节点中左侧的图标上，当鼠标变为+时按下鼠标移动到另一个节点中松开鼠标</div>
            </el-tab-pane>
            <el-tab-pane label="如何添加传递参数">
                <el-divider content-position="left">如何在组件之间添加传递参数</el-divider>
                <div class="info">点击画布中的连线，在页面右侧会出现一个表单，在from,to中分别选择参数，点击【保存】</div>
            </el-tab-pane>
            <el-tab-pane label="注意">
                <el-divider content-position="left">注意事项</el-divider>
                <div class="info">每次在右边修改了组件数据后,都需要点击下方的【保存】才能生效</div>
            </el-tab-pane> 

        </el-tabs>
    </el-dialog>
</template>



<script>
    export default {
        data() {
            return {
                dialogVisible: false
            }
        },
        components: {},
        methods: {
            init() {
                this.dialogVisible = true
            }
        }
    }
</script>

<style>
    .flowHelp {
        height: 80%;
    }
    .el-tabs--left .el-tabs__item.is-left {
        text-align: right;
        color: cadetblue;
    }
    .el-divider__text.is-left {
        left: 20px;
        -webkit-transform: translateY(-50%);
        transform: translateY(-50%);
        background-color: #005D3B;
        color:cadetblue;
    }
    .info{
        color: cornflowerblue;
        font-size: 1.02rem;
    }

</style>
