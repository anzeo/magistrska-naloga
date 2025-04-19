<script setup>
import { ref } from "vue";

defineProps({
  msg: String,
});

const count = ref(0);
const value = ref(0);
</script>

<template>
  <h1>{{ msg }}</h1>

  <h1 class="text-4xl">Hello world!</h1>

  <div class="card">
    <Button @click="count++">count is {{ count }}</Button>
    <p>
      Edit
      <code>components/HelloWorld.vue</code> to test HMR
    </p>
  </div>

  <InputText class="four-column-summary"></InputText>

  <p>
    Check out
    <a href="https://vuejs.org/guide/quick-start.html#local" target="_blank"
      >create-vue</a
    >, the official Vue + Vite starter
  </p>
  <p>
    Learn more about IDE Support for Vue in the
    <a
      href="https://vuejs.org/guide/scaling-up/tooling.html#ide-support"
      target="_blank"
      >Vue Docs Scaling up Guide</a
    >.
  </p>
  <p class="read-the-docs">Click on the Vite and Vue logos to learn more</p>
</template>

<style scoped></style>
