import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  retries: 0,
  workers: 1,
  outputDir: '../artifacts/agentic-ai/playwright/test-results',
  reporter: [
    ['line'],
    ['html', { outputFolder: '../artifacts/agentic-ai/playwright/report', open: 'never' }],
    ['junit', { outputFile: '../artifacts/agentic-ai/playwright/playwright-results.xml' }],
  ],
  use: {
    baseURL: 'http://127.0.0.1:4173',
    locale: 'en-US',
    trace: 'on',
    screenshot: 'only-on-failure',
    ...devices['Desktop Chrome'],
  },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 4173',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: true,
    timeout: 120000,
  },
})
