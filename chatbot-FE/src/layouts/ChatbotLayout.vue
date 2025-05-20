<template>
  <div id="chatbot-container">
    <div class="flex">
      <!--   Sidebar area    -->
      <div class="sidebar-container">
        <SidebarMenu></SidebarMenu>
      </div>

      <!-- Main content area -->
      <div class="content-container flex-1 h-screen">
        <router-view :key="$route.fullPath"></router-view>
      </div>
    </div>
  </div>
</template>

<script>
import SidebarMenu from "./SidebarMenu.vue";
import Chatbot from "../components/Chatbot.vue";

export default {
  name: "ChatbotLayout",

  components: {
    Chatbot,
    SidebarMenu,
  },

  data() {
    return {
      sidebarVisible: true,
    };
  },

  computed: {},

  mounted() {},

  methods: {},
};
</script>

<style scoped></style>
