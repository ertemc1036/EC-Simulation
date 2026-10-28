export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
  })
  const data = (await res.json().catch(() => ({}))) as { error?: string }
  if (!res.ok) {
    throw new Error(data.error || `Request failed (${res.status})`)
  }
  return data as T
}
