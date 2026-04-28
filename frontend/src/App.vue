<script setup>
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  NConfigProvider, NDialogProvider, NMessageProvider, NNotificationProvider,
  zhCN, dateZhCN, enUS, dateEnUS,
} from 'naive-ui'

const { locale } = useI18n()

const naiveLocale     = computed(() => (locale.value === 'zh-CN' ? zhCN     : enUS))
const naiveDateLocale = computed(() => (locale.value === 'zh-CN' ? dateZhCN : dateEnUS))

const themeOverrides = {
  common: {
    primaryColor: '#c47c3e',
    primaryColorHover: '#d4924f',
    primaryColorPressed: '#a86930',
    primaryColorSuppl: '#e5a876',
    fontFamily: "'Montserrat', -apple-system, sans-serif",
    fontFamilyMono: "'Fira Code', monospace",
    borderRadius: '9px',
    borderRadiusSmall: '7px',
  },
  Button: {
    borderRadiusMedium: '9px',
    borderRadiusSmall: '7px',
    fontWeightStrong: '600',
  },
  Input: {
    borderRadius: '10px',
  },
  Card: {
    borderRadius: '14px',
  },
  Modal: {
    borderRadius: '16px',
  },
  Tag: {
    borderRadius: '99px',
  },
}
</script>

<template>
  <NConfigProvider :theme-overrides="themeOverrides" :locale="naiveLocale" :date-locale="naiveDateLocale">
    <NMessageProvider>
      <NNotificationProvider>
        <NDialogProvider>
          <RouterView />
        </NDialogProvider>
      </NNotificationProvider>
    </NMessageProvider>
  </NConfigProvider>
</template>
