import { createRouter, createWebHistory } from "vue-router";
import ChatbotLayout from "../layouts/ChatbotLayout.vue";
import Chatbot from "../components/Chatbot.vue";

const routes = [
  {
    path: "/",
    name: "ChatbotLayout",
    redirect: { name: "Chat" },
    component: ChatbotLayout,
    children: [
      {
        path: ":chatId?",
        name: "Chat",
        component: Chatbot,
      },
    ],
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
