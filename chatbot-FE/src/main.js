import { createApp } from "vue";
import App from "./App.vue";
import Aura from "@primeuix/themes/aura";
import PrimeVue from "primevue/config";
import router from "./router";
import axios from "axios";
import VueAxios from "vue-axios";
import VueSweetalert2 from "vue-sweetalert2";

import "primeicons/primeicons.css";
import "./styles/app.css";
import "sweetalert2/dist/sweetalert2.min.css";
import { ConfirmationService, ToastService } from "primevue";

const app = createApp(App);

app.use(router);
app.use(VueAxios, axios);
app.use(ToastService);
app.use(ConfirmationService);
app.use(VueSweetalert2);

app.config.globalProperties.$config = window.config;
app.config.globalProperties.$axios = axios;

app.use(PrimeVue, {
  theme: {
    preset: Aura,
    options: {
      cssLayer: {
        name: "primevue",
        order: "theme, base, primevue",
      },
    },
  },
});

app.mount("#app");
