// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from "vue";
import App from "./App";
Vue.config.productionTip = false;

// elementui
import ElementUI from "element-ui";
import "element-ui/lib/theme-chalk/index.css";
import "@/assets/css/index.css";
Vue.use(ElementUI, { size: "small" });

// Contextmenu
import Contextmenu from "vue-contextmenujs";
Vue.use(Contextmenu);

// VueCodemirror
import VueCodemirror from "vue-codemirror";
import "codemirror/lib/codemirror.css";
import 'codemirror/mode/python/python.js'
import'codemirror/addon/selection/active-line.js'
import'codemirror/addon/edit/closebrackets.js'
import'codemirror/mode/clike/clike.js'
import'codemirror/addon/edit/matchbrackets.js'
import'codemirror/addon/comment/comment.js'
import'codemirror/addon/dialog/dialog.js'
import'codemirror/addon/dialog/dialog.css'
import'codemirror/addon/search/searchcursor.js'
import'codemirror/addon/search/search.js'
import'codemirror/keymap/sublime.js'
import 'codemirror/theme/base16-light.css'
import 'codemirror/theme/darcula.css'


Vue.use(VueCodemirror, {
  options: {
    autoCloseBrackets: true,
    tabSize: 4,
    styleActiveLine: true,
    lineNumbers: true,
    line: true,
    // mode: "text/x-python",
    mode: { name: "javascript", json: true },
    lineNumbers: true,

    // https://codemirror.net/5/demo/theme.html 
    theme: "darcula",
    // theme: "default",
    keyMap: "sublime"
  }
});

import {i18n} from '@/i18n/index.js'; 
Vue.use(i18n);


/* eslint-disable no-new */
new Vue({
  el: "#app",
  i18n, 
  components: { App },
  template: "<App/>"
});


