<template>
  <div class="flex flex-col h-full max-w-[1000px] mx-auto">
    <div class="chat-container flex-1 flex justify-center items-end mb-4">
      <div v-if="!chatId" class="new-chat-info-container text-center">
        <h2 class="text-2xl font-semibold">Kako ti lahko pomagam?</h2>
        <p class="text-sm text-gray-600">
          Sem pogovorni robot specializiran na področju Akta o umetni
          inteligenci.
        </p>
      </div>
    </div>

    <div class="query-input-container" :class="chatId ? 'mb-3' : 'flex-grow-1'">
      <div
        class="bg-white border rounded-2xl border-gray-400 p-4 flex items-center gap-3"
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
</template>

<script>
export default {
  name: "Chatbot",

  data() {
    return {
      chatId: null,
    };
  },

  mounted() {
    console.log("heree");
    if (this.$route.params.chatId) {
      this.chatId = this.$route.params.chatId;
    }
  },

  methods: {},
};
</script>

<style scoped></style>
