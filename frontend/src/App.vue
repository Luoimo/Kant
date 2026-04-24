<script setup>
import { NConfigProvider, NDialogProvider, NMessageProvider, NNotificationProvider } from 'naive-ui'

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
  <NConfigProvider :theme-overrides="themeOverrides">
    <NMessageProvider>
      <NNotificationProvider>
        <NDialogProvider>
          <RouterView />
        </NDialogProvider>
      </NNotificationProvider>
    </NMessageProvider>
  </NConfigProvider>
</template>
