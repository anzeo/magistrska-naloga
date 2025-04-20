<template>
  <div
    class="z-[1001] p-3 bg-gray-100 h-full shadow-xl flex-col transition-[width] duration-300 ease-in-out flex overflow-hidden"
    :class="visible ? 'w-[250px] md:relative absolute' : 'w-[64px] relative'"
  >
    <div class="flex flex-row items-center justify-end">
      <Button
        severity="secondary"
        icon="pi pi-bars"
        icon-class="text-[18px]"
        @click="visible = !visible"
      ></Button>
    </div>

    <div class="mt-2 mb-3">
      <Button
        severity="primary"
        class="w-full flex items-center justify-center"
        @click="handleNewChatBtnClick"
      >
        <Transition
          enter-active-class="transition-opacity duration-300 ease-in-out"
          enter-from-class="opacity-0"
          enter-to-class="opacity-100"
          leave-active-class="transition-opacity duration-300 ease-in-out"
          leave-from-class="opacity-100"
          leave-to-class="opacity-0"
        >
          <span v-if="visible" class="whitespace-nowrap overflow-hidden"
            >Nov pogovor</span
          >
        </Transition>
        <i class="pi pi-comments text-[18px]"></i>
      </Button>
    </div>

    <Transition
      enter-active-class="transition-all duration-200 ease-in-out"
      enter-from-class="opacity-0 translate-x-2"
      enter-to-class="opacity-100 translate-x-0"
      leave-active-class="transition-all duration-200 ease-in-out"
      leave-from-class="opacity-100 translate-x-0"
      leave-to-class="opacity-0 translate-x-2"
    >
      <div
        v-if="visible"
        class="flex-col text-sm flex overflow-scroll"
        style="flex: 1 1 0"
      >
        <div>
          <div v-for="chat in chats">
            <div
              class="px-2 py-2 rounded-lg hover:bg-gray-200 cursor-pointer truncate"
              :title="chat.label"
              @click="handleGoToChatBtnClick"
            >
              {{ chat.label }}
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </div>

  <!--  Backdrop  -->
  <div
    v-if="visible"
    class="fixed inset-0 bg-gray-500 opacity-30 z-[1000] md:hidden block"
    @click="visible = false"
  ></div>
</template>

<script>
import { Drawer } from "primevue";
import router from "../router/index.js";

export default {
  name: "Sidebar",

  components: { Drawer },

  props: {},

  data() {
    return {
      visible: true,

      chats: [
        {
          label: "Chat 1",
        },
        {
          label: "Chat 1 with very very long name that is truncated",
        },
        {
          label: "Chat 1",
        },
        {
          label: "Chat 1",
        },
        {
          label: "Chat 1",
        },
        {
          label: "Chat 1",
        },
        {
          label: "Chat 1",
        },
        {
          label: "Chat 1 with very very long name that is truncated",
        },
      ],
    };
  },

  methods: {
    async handleNewChatBtnClick() {
      await this.$router.push({ name: "Chat" });
    },

    async handleGoToChatBtnClick() {
      console.log("Go to chat");
      // await this.$router.push({ name: "Chat", params: { chatId: "abc" } });
    },
  },
};
</script>

<style scoped></style>
