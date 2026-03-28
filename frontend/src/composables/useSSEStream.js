// fetchSSEStream — shared SSE stream reader used by chat.js and ReaderChat.vue
// url: string, body: object
// callbacks: { onThinking, onStatus, onToken, onDone, onError }
export async function fetchSSEStream(url, body, { onThinking, onStatus, onToken, onDone, onError } = {}) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop()
    for (const part of parts) {
      const line = part.trim()
      if (!line.startsWith('data: ')) continue
      let evt
      try { evt = JSON.parse(line.slice(6)) } catch { continue }
      if (evt.type === 'thinking') onThinking?.()
      else if (evt.type === 'status') onStatus?.(evt.text)
      else if (evt.type === 'token') onToken?.(evt.text)
      else if (evt.type === 'done') onDone?.(evt)
      else if (evt.type === 'error') onError?.(evt.message)
    }
  }
}
