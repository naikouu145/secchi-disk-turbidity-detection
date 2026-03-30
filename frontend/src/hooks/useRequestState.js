import { useCallback, useState } from 'react'

const INITIAL_STATE = {
  status: 'idle',
  error: '',
  activeRequests: 0,
}

export function useRequestState() {
  const [requestState, setRequestState] = useState(INITIAL_STATE)

  const runRequest = useCallback(async (requestFn) => {
    setRequestState((prev) => ({
      ...prev,
      status: 'loading',
      error: '',
      activeRequests: prev.activeRequests + 1,
    }))

    try {
      const result = await requestFn()

      setRequestState((prev) => {
        const nextActive = Math.max(0, prev.activeRequests - 1)
        return {
          ...prev,
          status: nextActive > 0 ? 'loading' : 'success',
          error: '',
          activeRequests: nextActive,
        }
      })

      return result
    } catch (error) {
      setRequestState((prev) => {
        const nextActive = Math.max(0, prev.activeRequests - 1)
        return {
          ...prev,
          status: nextActive > 0 ? 'loading' : 'error',
          error: error?.message || 'Request failed',
          activeRequests: nextActive,
        }
      })
      throw error
    }
  }, [])

  const reset = useCallback(() => {
    setRequestState(INITIAL_STATE)
  }, [])

  return {
    ...requestState,
    isIdle: requestState.status === 'idle',
    isLoading: requestState.status === 'loading',
    isSuccess: requestState.status === 'success',
    isError: requestState.status === 'error',
    runRequest,
    reset,
  }
}
