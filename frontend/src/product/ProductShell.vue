<script setup lang="ts">
import type { ProductView } from './types'

const view = defineModel<ProductView>({ required: true })

const items: { id: ProductView; label: string }[] = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'projects', label: 'Projects' },
  { id: 'studies', label: 'Research Studies' },
  { id: 'audiences', label: 'Synthetic Audiences' },
  { id: 'lab', label: 'Simulation Lab' },
  { id: 'reports', label: 'Reports' },
]
</script>

<template>
  <div class="nc-frame">
    <aside class="nc-side">
      <div>
        <p class="nc-mark">NanoCrab Lab</p>
        <p class="nc-product">Research MVP 1.0</p>
      </div>
      <nav class="nc-nav" aria-label="Product">
        <button
          v-for="item in items"
          :key="item.id"
          type="button"
          :class="{ active: view === item.id }"
          :aria-current="view === item.id ? 'page' : undefined"
          @click="view = item.id"
        >
          {{ item.label }}
        </button>
      </nav>
      <p class="nc-side-note">
        Synthetic personas are not human research. Sessions on this MVP are offline
        fixtures, not genuine model output. Provider spend stays at zero.
      </p>
    </aside>
    <main class="nc-main" :class="{ 'nc-main-lab': view === 'lab' }">
      <slot />
    </main>
  </div>
</template>
