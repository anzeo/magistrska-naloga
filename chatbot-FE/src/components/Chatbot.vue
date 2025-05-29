<template>
  <div class="flex flex-col h-full w-full py-4">
    <div
      v-if="!chatId && !currentMessageData.isStreaming"
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
      v-else-if="(!chatId && currentMessageData.isStreaming) || chatId"
      class="chat-container -mb-4 overflow-hidden flex-1 flex justify-center w-full items-end"
    >
      <div class="relative h-full w-full">
        <div class="flex flex-1 h-full overflow-y-auto flex-col">
          <div class="pb-[150px] px-4">
            <div
              ref="scrollContainer"
              class="flex flex-1 h-full max-w-[1000px] mx-auto flex-col"
            >
              <div class="flex flex-col">
                <!-- CHAT HISTORY -->
                <div
                  v-for="(chat, index) in chatHistory"
                  class="flex flex-col"
                  :style="
                    index === chatHistory.length - 1 &&
                    !currentMessageData.isStreaming &&
                    chatbotInvoked
                      ? 'min-height: calc(100dvh - 250px)'
                      : ''
                  "
                >
                  <div class="flex items-start py-5">
                    <Avatar
                      icon="pi pi-user"
                      class="mr-2 mt-1 bg-primary-100"
                      shape="square"
                    />
                    <div class="max-w-7/12 flex">
                      <div class="rounded-xl bg-gray-100 px-4 py-2">
                        <div class="whitespace-pre-wrap text-base">
                          {{ chat.human.content }}
                        </div>
                      </div>
                    </div>
                  </div>
                  <div class="py-5">
                    <div class="flex items-start">
                      <Avatar
                        icon="pi pi-android"
                        class="mr-2 bg-secondary-100"
                        shape="square"
                      />
                      <div class="w-full">
                        <div class="px-2 pl-2">
                          <div class="whitespace-pre-wrap text-base">
                            {{ chat.ai.content }}
                          </div>
                        </div>

                        <div
                          v-if="chat.ai.relevant_part_texts?.length"
                          class="pt-4"
                        >
                          <Accordion multiple>
                            <AccordionPanel
                              v-for="(relevant_part, index) in chat.ai
                                .relevant_part_texts"
                              :value="index"
                            >
                              <AccordionHeader class="bg-gray-50 py-3.5">
                                <div class="flex flex-1">
                                  <span class="self-center">
                                    Referenca {{ index + 1 }}
                                  </span>
                                  <span
                                    class="text-xs ml-1.5 text-muted-color self-end"
                                    style="line-height: normal"
                                    >({{
                                      getReferenceTitle(relevant_part)
                                    }})</span
                                  >
                                </div>
                              </AccordionHeader>

                              <AccordionContent
                                class="relevant-passage-accordion-content"
                              >
                                <div class="text-end mb-1">
                                  <a
                                    :href="`${$config.api.aiActUrl}#${relevant_part?.id}`"
                                    target="_blank"
                                    style="line-height: normal"
                                    class="text-xs font-normal text-blue-600 dark:text-blue-500 hover:underline self-center"
                                    @click.stop
                                    >Odpri v dokumentu</a
                                  >
                                </div>
                                <Card class="rounded-sm">
                                  <template #content>
                                    <div
                                      v-html="
                                        getFormattedAndMarkedVsebina(
                                          relevant_part.full_content,
                                          relevant_part.text,
                                          `relevant_passage_${chat.ai.id}_${index}_${relevant_part.id}`
                                        )
                                      "
                                      :id="`relevant_passage_${chat.ai.id}_${index}_${relevant_part.id}`"
                                      class="font-mono whitespace-pre-wrap text-[13px]"
                                    ></div>
                                  </template>
                                </Card>
                              </AccordionContent>
                            </AccordionPanel>
                          </Accordion>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <!-- CURRENT MESSAGE -->
                <div
                  v-if="currentMessageData.isStreaming"
                  style="min-height: calc(100dvh - 250px)"
                >
                  <div class="flex items-start py-5">
                    <Avatar
                      icon="pi pi-user"
                      class="mr-2 mt-1 bg-primary-100"
                      shape="square"
                    />
                    <div class="max-w-7/12 flex">
                      <div class="rounded-xl bg-gray-100 px-4 py-2">
                        <div class="whitespace-pre-wrap text-base">
                          {{ currentMessageData.human }}
                        </div>
                      </div>
                    </div>
                  </div>
                  <div
                    class="flex items-start py-5"
                    ref="currentAIMessageContainer"
                  >
                    <Avatar
                      icon="pi pi-android"
                      class="mr-2 bg-secondary-100"
                      shape="square"
                    />
                    <div class="w-full">
                      <div class="px-2 pl-2">
                        <div
                          v-if="currentMessageData.aiStep !== null"
                          class="ai-step text-shimmer whitespace-pre-wrap text-base text-justify"
                        >
                          {{ currentMessageData.aiStep }}
                        </div>
                        <div v-else class="whitespace-pre-wrap text-base">
                          {{ currentMessageData.ai }}
                        </div>
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
      :class="
        (!chatId && currentMessageData.isStreaming) || chatId
          ? 'mb-3'
          : 'flex-grow-1'
      "
    >
      <div class="px-4">
        <div class="max-w-[1000px] mx-auto">
          <div
            class="bg-white border rounded-3xl border-gray-400 p-4 flex items-center gap-3 shadow-md"
          >
            <Textarea
              v-model.trim="userQuery"
              @keydown.enter.exact.prevent="sendMessage"
              rows="1"
              autoResize
              placeholder="Vnesite sporočilo..."
              class="flex-1 border-0 shadow-none"
              style="max-height: 200px"
            />

            <Button
              icon="pi pi-send"
              class="self-end"
              @click="sendMessage"
              :disabled="isUserQueryInvalid"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { SSE } from "sse.js";
import emitter from "../event-bus.js";
import { Accordion } from "primevue";
import Mark from "mark.js";

export default {
  name: "Chatbot",

  data() {
    return {
      chatId: null,

      chatHistory: [],

      userQuery: "",

      currentMessageData: {
        isStreaming: false,
        human: "",
        ai: "",
        aiStep: null,
      },

      chatbotInvoked: false,
    };
  },

  computed: {
    isUserQueryInvalid() {
      return ["", null].includes(this.userQuery);
    },
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

    resetCurrentMessageData() {
      this.currentMessageData = {
        isStreaming: false,
        human: "",
        ai: "",
        aiStep: null,
      };
    },

    async sendMessage() {
      if (this.isUserQueryInvalid) return;

      try {
        let url = `${this.$config.api.baseUrl}chatbot/invoke`;

        let body = {
          user_input: this.userQuery,
          ...(this.chatId ? { chat_id: this.chatId } : {}),
        };

        this.currentMessageData.human = this.userQuery;
        this.currentMessageData.aiStep = "Invoking chatbot";
        this.currentMessageData.isStreaming = true;
        this.userQuery = "";

        await this.$nextTick(() => {
          const container = this.$refs.scrollContainer;
          const messageDiv = this.$refs.currentAIMessageContainer;
          if (container && messageDiv) {
            messageDiv.scrollIntoView({
              behavior: "smooth",
              block: "start",
              inline: "nearest",
            });
          }
        });

        const isFirstMessage = this.chatId === null;
        let newChatData = null;

        const source = new SSE(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          payload: JSON.stringify(body),
        });

        const _this = this;

        source.addEventListener("message", async function (e) {
          const data = JSON.parse(e.data);
          switch (data.type) {
            case "step":
              _this.currentMessageData.aiStep = data["v"]["intermediate_step"];
              break;
            case "stream_complete":
              _this.chatHistory.push(JSON.parse(data["v"])["turn"]);
              _this.$forceUpdate();
              break;
            case "chat_data":
              if (isFirstMessage) {
                _this.chatId = data["v"]["id"];
                newChatData = data["v"];
              }
              break;
            default:
              console.log("Unknown event type:", data.type);
          }
        });

        source.addEventListener("answer", function (e) {
          const data = JSON.parse(e.data);
          _this.currentMessageData.ai += data["v"];
          _this.currentMessageData.aiStep = null;
        });

        source.addEventListener("readystatechange", async function (e) {
          // When closing the stream, you should expect:
          // A readystatechange event with a readyState of CLOSED (2);
          if (e.readyState === 2) {
            if (isFirstMessage) {
              emitter.emit("new-chat", newChatData);
              await _this.$router.replace({
                name: "Chat",
                params: { chatId: _this.chatId },
              });
            }

            _this.chatbotInvoked = true;
            _this.resetCurrentMessageData();
          }
        });

        source.addEventListener("error", async function (e) {
          const data = JSON.parse(e.data);
          console.error(e);

          _this.$swal({
            title: "Prišlo je do napake!",
            html: data.error || data,
            icon: "error",
            showCloseButton: true,
            showCancelButton: false,
            focusConfirm: false,
          });
        });
      } catch (error) {
        console.error(error);
        this.$swal({
          title: "Prišlo je do napake!",
          html: error.detail || error,
          icon: "error",
          showCloseButton: true,
          showCancelButton: false,
          focusConfirm: false,
        });
      }
    },

    getFormattedAndMarkedVsebina(full_content, relevant_parts, elementId) {
      if (!full_content) {
        return "";
      }

      let instance = new Mark(document.getElementById(elementId));
      instance.mark(relevant_parts, {
        value: "exactly",
        separateWordSearch: false,
      });

      if (full_content.id_elementa.includes("art_")) {
        // Check if this is a Člen (article)
        return `<div class="text-center"><i>Člen ${full_content.id_elementa.replace(
          "art_",
          ""
        )}</i></div>\n<div class="text-center font-bold">${
          full_content.naslov
        }</div>\n${full_content.vsebina}`;
      } else if (full_content.id_elementa.includes("rct_")) {
        // Check if this is a Točka (point)
        return `<div class="text-center"><i>(${full_content.id_elementa.replace(
          "rct_",
          ""
        )})</i></div>\n${full_content.vsebina}`;
      }
    },

    getReferenceTitle(relevant_part) {
      if (relevant_part.id.includes("art_")) {
        // Check if this is a Člen (article)
        return `Člen ${relevant_part.id.replace("art_", "")}`;
      } else if (relevant_part.id.includes("rct_")) {
        // Check if this is a Točka (point)
        return `Točka ${relevant_part.id.replace("rct_", "")}`;
      }
      return "Referenca";
    },
  },
};
</script>

<style>
.relevant-passage-accordion-content {
  .p-accordioncontent-content {
    background: var(--color-gray-50);
  }
}
</style>
