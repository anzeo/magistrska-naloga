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
        class="w-full flex items-center justify-center transition-all duration-300"
        :class="visible ? '' : 'gap-0'"
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
            >Nov klepet</span
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
            <div>
              <div
                class="px-2 py-2 rounded-lg cursor-pointer group"
                :class="{
                  'bg-gray-200':
                    visibleMenuId === chat.id ||
                    renamingChatData.chatId === chat.id,
                  'bg-gray-300': activeChatId === chat.id,
                  'hover:bg-gray-200': activeChatId !== chat.id,
                }"
                :title="chat.name"
                @click="handleGoToChatBtnClick(chat.id)"
              >
                <div class="flex items-center">
                  <template v-if="renamingChatData.chatId === chat.id">
                    <InputText
                      :ref="`renameInput_${chat.id}`"
                      class="h-[26px] my-[-3px] w-full"
                      size="small"
                      v-model="chat.name"
                      @click.stop
                      @keydown.enter="
                        $refs[`renameInput_${chat.id}`]?.[0]?.$el?.blur()
                      "
                      @blur="saveChatName(chat.id, chat.name)"
                    ></InputText>
                  </template>
                  <template v-else>
                    <span class="truncate-fade w-full">
                      {{ chat.name }}
                    </span>
                    <i
                      class="pi pi-ellipsis-h p-0.5 ml-2 opacity-70 transition-all duration-100 ease-in-out"
                      :class="{
                        '!flex':
                          visibleMenuId === chat.id || activeChatId === chat.id,
                        '!hidden':
                          visibleMenuId !== chat.id && activeChatId !== chat.id,
                        'group-hover:!flex': true,
                        'hover:opacity-100 hover:!font-bold': true,
                      }"
                      style="font-size: 14px; line-height: normal"
                      @click.stop="(e) => toggleActionsMenu(e, chat.id)"
                    ></i>
                    <Menu
                      :ref="`actionsMenu_${chat.id}`"
                      class="text-sm"
                      :model="actionsMenuItems(chat)"
                      @hide="
                        () => {
                          if (visibleMenuId === chat.id) visibleMenuId = null;
                        }
                      "
                      popup
                    >
                    </Menu>
                  </template>
                </div>
              </div>
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

  <Toast />
  <ConfirmDialog></ConfirmDialog>
</template>

<script>
import { Drawer } from "primevue";

export default {
  name: "Sidebar",

  components: { Drawer },

  props: {},

  data() {
    return {
      visible: true,

      activeChatId: null,

      visibleMenuId: null,
      renamingChatData: {},

      chats: [],
    };
  },

  watch: {
    $route(to, from) {
      if (to.params.chatId) {
        this.activeChatId = to.params.chatId;
      } else {
        this.activeChatId = null;
      }
    },
  },

  mounted() {
    if (this.$route.params.chatId) {
      this.activeChatId = this.$route.params.chatId;
    }
    this.getChats();
  },

  methods: {
    async getChats() {
      try {
        let url = `${this.$config.api.baseUrl}chats`;
        let resp = await this.$axios.get(url);

        this.chats = resp.data;
      } catch (error) {
        console.error(error);
        this.chats = [];
      }
    },

    async handleNewChatBtnClick() {
      await this.$router.push({ name: "Chat" });
    },

    async handleGoToChatBtnClick(chatId) {
      await this.$router.push({ name: "Chat", params: { chatId: chatId } });
    },

    toggleActionsMenu(event, chatId) {
      if (this.visibleMenuId && this.visibleMenuId !== chatId) {
        this.$refs[`actionsMenu_${this.visibleMenuId}`]?.[0]?.hide();
      }

      // Toggle current popover
      if (this.visibleMenuId === chatId) {
        this.visibleMenuId = null;
        this.$refs[`actionsMenu_${chatId}`]?.[0]?.hide();
      } else {
        this.visibleMenuId = chatId;
        this.$refs[`actionsMenu_${chatId}`]?.[0]?.toggle(event);
      }
    },

    actionsMenuItems(chat) {
      return [
        {
          label: "Preimenuj",
          icon: "pi pi-pencil",
          command: () => this.startEditingName(chat),
        },
        { separator: true },
        {
          label: "Izbriši",
          icon: "pi pi-trash",
          class: "menu-item-danger",
          command: () => this.deleteChat(chat.id),
        },
      ];
    },

    startEditingName(chat) {
      this.visibleMenuId = null;
      this.renamingChatData = {
        chatId: chat.id,
        initialChatName: chat.name,
      };
      this.$nextTick(() => {
        this.$refs[`renameInput_${chat.id}`]?.[0]?.$el?.focus();
      });
    },

    async saveChatName(chatId, name) {
      if (this.renamingChatData.initialChatName === name) {
        this.renamingChatData = {};
        return;
      }
      this.renamingChatData = {};
      await this.$axios
        .put(`${this.$config.api.baseUrl}chats/${chatId}`, { name: name })
        .then(async () => {
          await this.getChats();
          this.$toast.add({
            severity: "success",
            summary: "Akcija uspešna",
            detail: "Ime klepeta spremenjeno!",
            life: 3000,
          });
        })
        .catch((error) => {
          console.error(error);
        });
    },

    async deleteChat(chatId) {
      this.visibleMenuId = null;
      this.$confirm.require({
        message: "Ali ste prepričani, da želite izbrisati klepet?",
        header: "Potrditev brisanja",
        icon: "pi pi-info-circle",
        rejectProps: {
          label: "Prekliči",
          severity: "secondary",
          outlined: true,
        },
        acceptProps: {
          label: "Izbriši",
          severity: "danger",
        },
        accept: async () => {
          if (this.activeChatId === chatId) {
            await this.handleNewChatBtnClick();
          }
          await this.$axios
            .delete(`${this.$config.api.baseUrl}chats/${chatId}`)
            .then(async () => {
              await this.getChats();
              this.$toast.add({
                severity: "success",
                summary: "Akcija uspešna",
                detail: "Klepet izbrisan!",
                life: 3000,
              });
            })
            .catch((error) => {
              console.error(error);
            });
        },
        reject: () => {},
      });
    },
  },
};
</script>

<style>
.menu-item-danger {
  .p-menu-item-icon {
    color: #ff595c !important;
  }
  .p-menu-item-label {
    color: #ff595c !important;
  }
}
</style>
