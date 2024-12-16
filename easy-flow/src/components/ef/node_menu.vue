<template>
    <div class="flow-menu" ref="tool">
        <div v-for="menu  in  menuList" :key="menu.id">
            <span class="ef-node-pmenu" @click="menu.open = !menu.open"><i
                    :class="{ 'el-icon-caret-bottom': menu.open, 'el-icon-caret-right': !menu.open }"></i>&nbsp;{{
            menu.name }}</span>
            <ul v-show="menu.open" class="ef-node-menu-ul">
                <draggable @end="end" @start="move" v-model="menu.children" :options="draggableOptions">
                    <li v-for="subMenu in menu.children" class="ef-node-menu-li" :key="subMenu.id" :type="subMenu.type" :name="subMenu.name"
                        :parenttype="menu.type" @contextmenu.prevent="onContextmenu">
                        <i :class="subMenu.ico"></i> {{ subMenu.name }}
                    </li>
                </draggable>
            </ul>
        </div>

        <el-dialog  :visible.sync="dialogVisible_editerCode" width="90%" :append-to-body=true 
            :before-close="handleClose">
            <codemirror v-model="code"></codemirror>
            <span slot="footer" class="dialog-footer">
                <el-button @click="dialogVisible_editerCode = false" class="my-el-button">{{$t('common.return')}}</el-button>
                <el-button type="primary" @click="setCode"  class="my-el-button">{{ $t('common.ok') }}</el-button>
            </span>
        </el-dialog>



    </div>
</template>

<style>
.CodeMirror {
    height: 70vh;
} 

</style>

<script>
import { rootUrl } from '@/assets/js/host'
import draggable from 'vuedraggable'
import panel from '@/components/ef/panel'
import * as utils from '@/assets/js/utils'

var mousePosition = {
    left: -1,
    top: -1
}

export default {
    components: {
        panel
    },

    data() {
        return {
            activeNames: '1',
            // draggable配置参数参考 https://www.cnblogs.com/weixin186/p/10108679.html
            draggableOptions: {
                preventOnFilter: false,
                sort: false,
                disabled: false,
                ghostClass: 'tt',
                // 不使用H5原生的配置
                forceFallback: true,
                // 拖拽的时候样式
                // fallbackClass: 'flow-node-draggable'
            },
            // 默认打开的左侧菜单的id
            defaultOpeneds: [],
            menuList: [],
            nodeMenu: {},
            dialogVisible_editerCode: false,
            code: "",
            codeFilePath: ""
        }
    },
    components: {
        draggable
    },
    created() {
        /**
         * 以下是为了解决在火狐浏览器上推拽时弹出tab页到搜索问题
         * @param event
         */
        if (this.isFirefox()) {
            document.body.ondrop = function (event) {
                // 解决火狐浏览器无法获取鼠标拖拽结束的坐标问题
                mousePosition.left = event.layerX
                mousePosition.top = event.clientY - 50
                event.preventDefault();
                event.stopPropagation();
            }
        }
        this.getInitConfigData();
    },
    methods: {
        handleClose(done) {
            this.$confirm('Confirm closure?')
                .then(_ => {
                    done();
                })
                .catch(_ => { });
        },

        onContextmenu(event) {
            this.$contextmenu({
                items: [
                    {
                        label: this.$t('nodemenu.edit_code'),
                        icon: "el-icon-edit",
                        onClick: () => {
                            let thisNode = null;
                            for (let i in this.menuList) {
                                const closestLiElement = event.target.closest('li');
                                const parentTypeValue = closestLiElement.getAttribute('parenttype');
                                const typeValue = event.target.type;
                                if (this.menuList[i]['type'] == parentTypeValue) {
                                    for (let j in this.menuList[i]['children']) {
                                        if (this.menuList[i]['children'][j]['type'] == typeValue) {
                                            thisNode = this.menuList[i]['children'][j];
                                            break;
                                        }
                                    }
                                }
                            }
                            if (!"codeFilePath" in thisNode || thisNode.codeFilePath == undefined || thisNode.codeFilePath == "") {
                                this.$message({ message: "Component path not configured", type: "warning", duration: 2000 });
                                return
                            }
                            this.code = utils.do_getComponentCodeByFilePath(this, thisNode.codeFilePath);
                            this.codeFilePath = thisNode.codeFilePath;
                            this.dialogVisible_editerCode = true;
                        }

                    },
                    {
                        label: this.$t('nodemenu.del_com'),
                        icon: "el-icon-delete",
                        onClick: () => {
                            // console.log("删除此组件");
                            // 在全局配置文件中删除组件
                            let delNode = null;
                            for (let i in this.menuList) {
                                const closestLiElement = event.target.closest('li');
                                const parentTypeValue = closestLiElement.getAttribute('parenttype');
                                const typeValue = event.target.type;
                                if (this.menuList[i]['type'] == parentTypeValue) {
                                    for (let j in this.menuList[i]['children']) {
                                        if (this.menuList[i]['children'][j]['type'] == typeValue) {
                                            // 删除这个元素
                                            delNode = this.menuList[i]['children'][j];
                                            delNode['parent_type'] = parentTypeValue;
                                            this.menuList[i]['children'].splice(j, 1);
                                        }
                                    }
                                }
                            }
                            utils.do_saveConfigData(this, this.menuList,'del',delNode);
                            // 刷新
                            window.location.reload();
                        }

                    },
                ],
                event, // 鼠标事件信息
                customClass: "custom-class", // 自定义菜单 class
                zIndex: 3, // 菜单样式 z-index
                minWidth: 230 // 主菜单最小宽度
            });
            return false;
        },
        // 获取初始配置文件
        getInitConfigData() {
            this.menuList = [].concat(utils.do_getInitConfigData(this));
        },

        // 修改组件代码
        setCode() {
            utils.do_setComponentCodeByFilePath(this, this.codeFilePath, this.code);
            this.dialogVisible_editerCode = false;
            this.code = "";
            this.codeFilePath = "";
        },


        // 根据类型 ,名称 获取左侧菜单对象
        getMenuByType(type,name) {
            for (let i = 0; i < this.menuList.length; i++) {
                let children = this.menuList[i].children;
                for (let j = 0; j < children.length; j++) {
                    if (children[j].type === type && children[j].name === name) {
                        return children[j]
                    }
                }
            }
        },
        // 拖拽开始时触发
        move(evt, a, b, c) {
            let type = evt.item.attributes.type.nodeValue
            let name = evt.item.attributes.name.nodeValue
            this.nodeMenu = this.getMenuByType(type,name)
        },
        // 拖拽结束时触发
        end(evt, e) {
            this.$emit('addNode', evt, this.nodeMenu, mousePosition)
        },
        // 是否是火狐浏览器
        isFirefox() {
            var userAgent = navigator.userAgent
            if (userAgent.indexOf("Firefox") > -1) {
                return true
            }
            return false
        }
    }
}
</script>
