<template>
  <div class="flex flex-col h-full w-full py-4">
    <div
      v-if="!chatId"
      class="overflow-hidden flex-1 flex justify-center items-end"
    >
      <div class="px-4">
        <div class="new-chat-info-container text-center flex-1">
          <h2 class="text-2xl font-semibold">Kako ti lahko pomagam?</h2>
          <p class="text-sm text-gray-600 mb-1">
            Sem pogovorni robot specializiran na področju Akta o umetni
            inteligenci.
          </p>
        </div>
      </div>
    </div>

    <div
      v-else
      class="chat-container -mb-4 overflow-hidden flex-1 flex justify-center w-full items-end"
    >
      <div class="relative h-full w-full">
        <div class="flex flex-1 h-full overflow-y-auto flex-col">
          <div class="pb-[150px] px-4">
            <div class="flex flex-1 h-full max-w-[1000px] mx-auto flex-col">
              <div class="flex flex-col">
                <div v-for="chat in chatHistory" class="flex flex-col">
                  <div class="max-w-7/12 self-end py-5">
                    <div class="rounded-2xl bg-gray-100 px-4 py-3">
                      <div class="whitespace-pre-wrap text-base">
                        {{ chat.human.content }}
                      </div>
                    </div>
                  </div>
                  <div class="w-full py-5">
                    <div class="">
                      <div class="whitespace-pre-wrap text-base text-justify">
                        {{ chat.ai.content }}
                      </div>
                    </div>
                  </div>
                </div>
                <div ref="bottomAnchor"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div
      class="query-input-container relative"
      :class="chatId ? 'mb-3' : 'flex-grow-1'"
    >
      <div class="px-4">
        <div class="max-w-[1000px] mx-auto">
          <div
            class="bg-white border rounded-3xl border-gray-400 p-4 flex items-center gap-3 shadow-md"
          >
            <Textarea
              rows="1"
              autoResize
              placeholder="Vnesite sporočilo..."
              class="flex-1 border-0 shadow-none"
              style="max-height: 200px"
            />

            <Button icon="pi pi-send" class="self-end" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "Chatbot",

  data() {
    return {
      chatId: null,

      chatHistory: [],
    };
  },

  async mounted() {
    if (this.$route.params.chatId) {
      this.chatId = this.$route.params.chatId;
      await this.getChatHistory();
      await this.$nextTick(() => {
        this.$refs.bottomAnchor?.scrollIntoView({ behavior: "instant" });
      });
    }
  },

  methods: {
    async getChatHistory() {
      try {
        let url = `${this.$config.api.baseUrl}chat-history/${this.chatId}`;
        let resp = await this.$axios.get(url);

        this.chatHistory = resp.data;
      } catch (error) {
        console.error(error);
        this.chatHistory = [];
      }
    },
  },
};
</script>

<style scoped></style>
