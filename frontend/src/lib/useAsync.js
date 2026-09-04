import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Fetch helper that holds the previous render while refetching, so switching
 * ticker or range dims the card instead of flashing a skeleton.
 */
export function useAsync(fn, deps = [], { enabled = true } = {}) {
  const [state, setState] = useState({ data: null, error: null, loading: enabled })
  const [nonce, setNonce] = useState(0)
  const latest = useRef(0)

  const run = useCallback(() => setNonce((n) => n + 1), [])

  useEffect(() => {
    if (!enabled) { setState((s) => ({ ...s, loading: false })); return }
    const id = ++latest.current
    setState((s) => ({ ...s, loading: true, error: null }))
    Promise.resolve()
      .then(fn)
      .then((data) => { if (id === latest.current) setState({ data, error: null, loading: false }) })
      .catch((error) => { if (id === latest.current) setState((s) => ({ data: s.data, error, loading: false })) })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce, enabled])

  return { ...state, refresh: run }
}
