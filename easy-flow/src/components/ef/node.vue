<template>

    <!-- 如果不是控制节点,就正常展示 -->
    <div v-if="!node.type.includes('Controller')" ref="node" :style="nodeContainerStyle" @click="clickNode"   
        @mouseup="changeNodeSite" :class="nodeContainerClass">
        <!-- 最左侧的那条竖线 -->
        <div class="ef-node-left"></div>
        <!-- 节点类型的图标 -->
        <div class="ef-node-left-ico flow-node-drag">
            <i :class="nodeIcoClass"></i>
        </div>
        <!-- 节点名称 -->
        <div class="ef-node-text" :show-overflow-tooltip="true" >
            {{ node.name }}
        </div>

        <div :style="nodeContainerStyle"  class="nodeDescription" style="width: 100%;" >
            {{ node.description }}
        </div>



        <!-- 节点状态图标 -->
        <div class="ef-node-right-ico">
            <i class="el-icon-circle-check el-node-state-success" v-show="node.state === 'success'"></i>
            <i class="el-icon-circle-close el-node-state-error" v-show="node.state === 'error'"></i>
            <i class="el-icon-warning-outline el-node-state-warning" v-show="node.state === 'warning'"></i>
            <i class="el-icon-loading el-node-state-running" v-show="node.state === 'running'"></i>
        </div>
    </div>

    <!-- 如果是控制节点 就特殊渲染 -->


    <div v-else ref="node" @click="clickNode" @mouseup="changeNodeSite" :style="nodeContainerStyle" style="justify-content: center; "  
        :class="nodeContainerClass">

        <div class="ef-node-left" style="background-color: green;"></div>
        <!-- 节点类型的图标 -->
        <div class="ef-node-left-ico flow-node-drag">
            <i :class="nodeIcoClass"></i>
        </div>

        <div class="ef-node-text" :show-overflow-tooltip="true"  >
            {{ node.name }}
        </div>

        <div class="ef-node-right" style="background-color: green;"></div>

        <div style="width: 100%; display: flex; justify-content: center; margin-bottom:8px;">
            <div class="arrow"></div>
        </div>

        <div :style="nodeContainerStyle" style="display: flex;flex-wrap: wrap;">
            <div v-for="component in node.components" :key="component" class="item">
                {{ component }}
            </div>
        </div>

        <div :style="nodeContainerStyle"  class="nodeDescription" style="width: 100%;" >
            {{ node.description }}
        </div>

    </div>


</template>

<style>
.nodeDescription{
    color:cadetblue;
    margin: 5px 10px 5px 10px;
}

</style>


<script>
export default {
    props: {
        node: Object,
        activeElement: Object
    },
    data() {
        return {}
    },
    computed: {
        nodeContainerClass() {
            return {
                'ef-node-container': true,
                'ef-node-active': this.activeElement.type == 'node' ? this.activeElement.nodeId === this.node.id : false
            }
        },
        // 节点容器样式
        nodeContainerStyle() {
            return {
                top: this.node.top,
                left: this.node.left
            }
        },
        nodeIcoClass() {
            var nodeIcoClass = {}
            nodeIcoClass[this.node.ico] = true
            // 添加该class可以推拽连线出来，viewOnly 可以控制节点是否运行编辑
            nodeIcoClass['flow-node-drag'] = this.node.viewOnly ? false : true
            return nodeIcoClass
        }
    },
    methods: {
        // 点击节点
        clickNode() {
            console.log(this.node);
            this.$emit('clickNode', this.node.id)
        },
        // 鼠标移动后抬起
        changeNodeSite() {
            // 避免抖动
            if (this.node.left == this.$refs.node.style.left && this.node.top == this.$refs.node.style.top) {
                return;
            }
            this.$emit('changeNodeSite', {
                nodeId: this.node.id,
                left: this.$refs.node.style.left,
                top: this.$refs.node.style.top,
            })
        },
    }
}
</script>
