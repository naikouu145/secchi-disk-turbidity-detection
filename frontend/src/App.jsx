import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  assessBatch,
  assessImage,
  checkHealth,
  getConfig,
  updateConfig,
} from './services/apiService'
import { useRequestState } from './hooks/useRequestState'

const MAX_FILE_SIZE = 10 * 1024 * 1024
const ACCEPTED_FILE_TYPES = 'JPG, PNG, WebP'

const SourcePieChart = lazy(() =>
  import('./components/common/ResultsCharts').then((module) => ({ default: module.SourcePieChart })),
)
const FeatureBarChart = lazy(() =>
  import('./components/common/ResultsCharts').then((module) => ({ default: module.FeatureBarChart })),
)
const ProbabilityBarChart = lazy(() =>
  import('./components/common/ResultsCharts').then((module) => ({ default: module.ProbabilityBarChart })),
)

function useDebouncedValue(value, delayMs) {
  const [debounced, setDebounced] = useState(value)

  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delayMs)
    return () => window.clearTimeout(timer)
  }, [value, delayMs])

  return debounced
}

function LoadingSkeleton({ className = '' }) {
  return <div className={`animate-pulse rounded-lg bg-sage-200/70 ${className}`} aria-hidden="true" />
}

function ChartFallback({ heightClass = 'h-48' }) {
  return (
    <div className={`rounded-lg border border-sage-200 bg-white p-3 ${heightClass}`}>
      <LoadingSkeleton className="h-full w-full" />
    </div>
  )
}

function getFriendlyErrorInfo(errorText, result) {
  const message = String(errorText || '').toLowerCase()

  if (result && result.disk_detected === false) {
    return {
      title: 'Secchi Disk Not Detected',
      description: 'The model could not confidently find a Secchi disk in this image.',
      suggestions: [
        'Use an image where the Secchi disk is centered and clearly visible.',
        'Avoid glare, heavy reflection, and severe motion blur.',
        'Try another angle or better lighting.',
      ],
    }
  }

  if (message.includes('invalid') && (message.includes('format') || message.includes('type') || message.includes('image'))) {
    return {
      title: 'Unsupported Image Format',
      description: `Please upload one of the supported formats: ${ACCEPTED_FILE_TYPES}.`,
      suggestions: ['Convert the image to JPG, PNG, or WebP and try again.'],
    }
  }

  if (message.includes('network') || message.includes('cannot reach') || message.includes('connection') || message.includes('timed out')) {
    return {
      title: 'Network Connection Issue',
      description: 'The app could not connect to the backend service.',
      suggestions: [
        'Check your internet or local network connection.',
        'Verify the backend server is running.',
        'Use Retry to attempt the request again.',
      ],
    }
  }

  if (message.includes('backend') || message.includes('500') || message.includes('503') || message.includes('server')) {
    return {
      title: 'Backend Processing Error',
      description: 'The server had trouble processing this request.',
      suggestions: ['Wait a moment and retry.', 'If this keeps happening, try a different image.'],
    }
  }

  return {
    title: 'Unable To Process Image',
    description: 'Something unexpected happened while analyzing the upload.',
    suggestions: ['Retry the request.', 'Upload a different image.', 'Refresh backend status and try again.'],
  }
}

async function createOptimizedPreview(file, maxDimension = 1400, quality = 0.82) {
  const fallbackUrl = URL.createObjectURL(file)

  return new Promise((resolve) => {
    const image = new Image()
    image.onload = () => {
      const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight))
      const width = Math.max(1, Math.round(image.naturalWidth * scale))
      const height = Math.max(1, Math.round(image.naturalHeight * scale))

      const canvas = document.createElement('canvas')
      canvas.width = width
      canvas.height = height

      const context = canvas.getContext('2d')
      if (!context) {
        resolve({ previewUrl: fallbackUrl, width: image.naturalWidth, height: image.naturalHeight })
        URL.revokeObjectURL(image.src)
        return
      }

      context.drawImage(image, 0, 0, width, height)

      canvas.toBlob(
        (blob) => {
          URL.revokeObjectURL(image.src)

          if (!blob) {
            resolve({ previewUrl: fallbackUrl, width: image.naturalWidth, height: image.naturalHeight })
            return
          }

          URL.revokeObjectURL(fallbackUrl)
          resolve({
            previewUrl: URL.createObjectURL(blob),
            width: image.naturalWidth,
            height: image.naturalHeight,
          })
        },
        'image/jpeg',
        quality,
      )
    }

    image.onerror = () => {
      URL.revokeObjectURL(image.src)
      resolve({ previewUrl: fallbackUrl, width: null, height: null })
    }

    image.src = fallbackUrl
  })
}

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes)) {
    return 'N/A'
  }
  if (bytes < 1024) {
    return `${bytes} B`
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function summarizeBatch(results) {
  const counts = {}
  results.forEach((entry) => {
    const category = entry.turbidity_category || 'Unknown'
    counts[category] = (counts[category] || 0) + 1
  })
  return Object.entries(counts).sort((a, b) => b[1] - a[1])
}

function normalizeSourceInfo(source) {
  if (!source) {
    return { primary_source: 'unknown', confidence: 0 }
  }

  if (typeof source === 'string') {
    return { primary_source: source, confidence: 0 }
  }

  return {
    primary_source: source.primary_source || 'unknown',
    confidence: Number(source.confidence ?? 0),
    algal_score: Number(source.algal_score ?? 0),
    sediment_score: Number(source.sediment_score ?? 0),
    note: source.note || '',
  }
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max)
}

function titleCase(value) {
  if (!value || typeof value !== 'string') {
    return 'Unknown'
  }

  return value
    .replace(/[_-]/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1).toLowerCase())
    .join(' ')
}

function getSeverityConfig(category) {
  if (category === 'High Turbidity' || category === 'Very High Turbidity') {
    return {
      label: 'Critical Turbidity',
      badgeClass: 'bg-red-100 text-red-700',
      barClass: 'bg-red-500',
      level: 95,
    }
  }

  if (category === 'Moderately Turbid') {
    return {
      label: 'Moderate Turbidity',
      badgeClass: 'bg-amber-100 text-amber-800',
      barClass: 'bg-amber-500',
      level: 65,
    }
  }

  if (category === 'Slightly Turbid') {
    return {
      label: 'Mild Turbidity',
      badgeClass: 'bg-sky-100 text-sky-700',
      barClass: 'bg-sky-500',
      level: 35,
    }
  }

  if (category === 'Clear Water') {
    return {
      label: 'Low Turbidity',
      badgeClass: 'bg-emerald-100 text-emerald-700',
      barClass: 'bg-emerald-500',
      level: 10,
    }
  }

  return {
    label: 'Unclassified',
    badgeClass: 'bg-slate-100 text-slate-700',
    barClass: 'bg-slate-400',
    level: 20,
  }
}

function sourceDescription(sourceType) {
  if (sourceType === 'algal') {
    return 'Algal-dominated turbidity often indicates nutrient enrichment and bloom dynamics.'
  }
  if (sourceType === 'sediment') {
    return 'Sediment-driven turbidity is commonly linked to erosion, runoff, or disturbance events.'
  }
  if (sourceType === 'mixed') {
    return 'Mixed-source turbidity combines biological and mineral/particulate influences.'
  }
  return 'Source type could not be confidently identified for this sample.'
}

function toPercent(value, digits = 1) {
  const safe = Number(value)
  if (!Number.isFinite(safe)) {
    return '0.0%'
  }
  return `${(safe * 100).toFixed(digits)}%`
}

function buildSourceChartData(sourceInfo) {
  const algal = clamp(Number(sourceInfo.algal_score ?? 0), 0, 1)
  const sediment = clamp(Number(sourceInfo.sediment_score ?? 0), 0, 1)
  const mixed = clamp(1 - Math.max(algal, sediment), 0, 1)

  if (algal === 0 && sediment === 0) {
    return [
      { name: 'Algal', value: sourceInfo.primary_source === 'algal' ? 1 : 0.25 },
      { name: 'Sediment', value: sourceInfo.primary_source === 'sediment' ? 1 : 0.25 },
      { name: 'Mixed', value: sourceInfo.primary_source === 'mixed' ? 1 : 0.25 },
    ]
  }

  return [
    { name: 'Algal', value: algal },
    { name: 'Sediment', value: sediment },
    { name: 'Mixed', value: mixed },
  ]
}

function buildFeatureChartData(featureContributions) {
  if (!featureContributions || typeof featureContributions !== 'object') {
    return []
  }

  return Object.entries(featureContributions)
    .map(([name, value]) => ({ name: titleCase(name), value: Number(value ?? 0) }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 8)
}

function buildProbabilityData(probabilities) {
  if (!probabilities || typeof probabilities !== 'object') {
    return []
  }

  return Object.entries(probabilities)
    .map(([name, value]) => ({ name, value: Number(value ?? 0) }))
    .filter((entry) => entry.value > 0)
    .sort((a, b) => b.value - a.value)
}

function downloadBlob(filename, mimeType, content) {
  const blob = new Blob([content], { type: mimeType })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(link.href)
}

function drawAnnotatedImage(previewUrl, analysis) {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.crossOrigin = 'anonymous'

    image.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = image.naturalWidth
      canvas.height = image.naturalHeight

      const context = canvas.getContext('2d')
      if (!context) {
        reject(new Error('Unable to render annotation canvas'))
        return
      }

      context.drawImage(image, 0, 0, canvas.width, canvas.height)

      const bbox = analysis?.bbox
      if (Array.isArray(bbox) && bbox.length === 4) {
        const [x1, y1, x2, y2] = bbox.map((value) => Number(value ?? 0))
        const width = Math.max(x2 - x1, 1)
        const height = Math.max(y2 - y1, 1)

        context.lineWidth = 4
        context.strokeStyle = '#0f766e'
        context.strokeRect(x1, y1, width, height)

        const label = `${analysis?.turbidity_category || 'Unknown'} | ${Number(analysis?.visibility_score ?? 0).toFixed(3)}`
        context.font = '600 20px ui-sans-serif'
        const textWidth = context.measureText(label).width
        context.fillStyle = '#0f766e'
        context.fillRect(x1, Math.max(0, y1 - 30), textWidth + 18, 30)
        context.fillStyle = '#ffffff'
        context.fillText(label, x1 + 9, Math.max(20, y1 - 9))
      }

      resolve(canvas.toDataURL('image/png'))
    }

    image.onerror = () => reject(new Error('Unable to load image for annotation'))
    image.src = previewUrl
  })
}

function App() {
  const [status, setStatus] = useState({ initialized: false, model_path: null })
  const {
    runRequest: runStatusRequest,
    isLoading: isStatusLoading,
    isError: isStatusError,
    error: statusError,
    activeRequests: statusActiveRequests,
  } = useRequestState()

  const {
    runRequest: runPredictRequest,
    isLoading: isPredictLoading,
    isError: isPredictError,
    error: predictError,
    activeRequests: predictActiveRequests,
  } = useRequestState()

  const {
    runRequest: runConfigRequest,
    isLoading: isConfigLoading,
    isError: isConfigError,
    error: configError,
  } = useRequestState()

  const [uploadedItems, setUploadedItems] = useState([])
  const [uploadMode, setUploadMode] = useState('single')

  const [selectedStandard, setSelectedStandard] = useState('auto')
  const [selectedWeightingMethod, setSelectedWeightingMethod] = useState('balanced')
  const [adaptiveScoring, setAdaptiveScoring] = useState(false)
  const [showDetailedOutput, setShowDetailedOutput] = useState(true)
  const [showSourceBreakdown, setShowSourceBreakdown] = useState(true)

  const [formError, setFormError] = useState('')
  const [dropzoneError, setDropzoneError] = useState('')
  const [singleResult, setSingleResult] = useState(null)
  const [batchResults, setBatchResults] = useState([])
  const [batchSummary, setBatchSummary] = useState([])
  const [resultsTab, setResultsTab] = useState('overview')
  const [imageZoom, setImageZoom] = useState(1)
  const debouncedImageZoom = useDebouncedValue(imageZoom, 100)
  const [comparisonSnapshots, setComparisonSnapshots] = useState([])
  const [exportMessage, setExportMessage] = useState('')
  const [toasts, setToasts] = useState([])
  const [uploadProgress, setUploadProgress] = useState(0)
  const [processStageText, setProcessStageText] = useState('Idle')
  const [lastAttemptMode, setLastAttemptMode] = useState('single')

  const replaceUploadedItems = useCallback((items) => {
    setUploadedItems((previous) => {
      previous.forEach((item) => URL.revokeObjectURL(item.previewUrl))
      return items
    })
  }, [])

  const appendUploadedItems = useCallback((items) => {
    setUploadedItems((previous) => [...previous, ...items])
  }, [])

  const refreshStatus = useCallback(async () => {
    try {
      const [health, config] = await runStatusRequest(() =>
        Promise.all([checkHealth(), getConfig()]),
      )

      setStatus({
        initialized: Boolean(health?.system_initialized),
        model_path: config?.model_path ?? null,
      })

      if (typeof config?.default_standard === 'string') {
        setSelectedStandard(config.default_standard)
      }
      if (typeof config?.default_weighting_method === 'string') {
        setSelectedWeightingMethod(config.default_weighting_method)
      }
      if (typeof config?.default_adaptive_scoring === 'boolean') {
        setAdaptiveScoring(config.default_adaptive_scoring)
      }
    } catch {
      setStatus({ initialized: false, model_path: null })
    }
  }, [runStatusRequest])

  useEffect(() => {
    const timerId = window.setTimeout(() => {
      void refreshStatus()
    }, 0)

    return () => window.clearTimeout(timerId)
  }, [refreshStatus])

  useEffect(
    () => () => {
      uploadedItems.forEach((item) => URL.revokeObjectURL(item.previewUrl))
    },
    [uploadedItems],
  )

  useEffect(() => {
    if (!isPredictLoading) {
      return undefined
    }

    setProcessStageText('Processing image analysis')
    const timer = window.setInterval(() => {
      setUploadProgress((previous) => {
        if (previous >= 92) {
          return previous
        }
        return previous + 3
      })
    }, 300)

    return () => window.clearInterval(timer)
  }, [isPredictLoading])

  useEffect(() => {
    if (!isPredictLoading && uploadProgress >= 100) {
      const timer = window.setTimeout(() => {
        setUploadProgress(0)
        setProcessStageText('Idle')
      }, 700)
      return () => window.clearTimeout(timer)
    }
    return undefined
  }, [isPredictLoading, uploadProgress])

  const pushToast = useCallback((type, message) => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`
    setToasts((previous) => [...previous, { id, type, message }])

    window.setTimeout(() => {
      setToasts((previous) => previous.filter((toast) => toast.id !== id))
    }, 4200)
  }, [])

  function handleUploadModeChange(nextMode) {
    if (nextMode === uploadMode) {
      return
    }

    setUploadMode(nextMode)
    setSingleResult(null)
    setBatchResults([])
    setBatchSummary([])
    setFormError('')
    setDropzoneError('')
    setResultsTab('overview')
    setExportMessage('')
    replaceUploadedItems([])
  }

  const onDrop = useCallback(
    async (acceptedFiles, fileRejections) => {
      setFormError('')

      if (fileRejections.length > 0) {
        const firstRejection = fileRejections[0]
        const firstError = firstRejection?.errors?.[0]

        if (firstError?.code === 'file-too-large') {
          setDropzoneError('File is too large. Maximum allowed size is 10 MB.')
          pushToast('error', 'Upload rejected: file exceeds 10 MB limit.')
        } else if (firstError?.code === 'file-invalid-type') {
          setDropzoneError(`Invalid file type. Accepted formats: ${ACCEPTED_FILE_TYPES}.`)
          pushToast('error', `Unsupported format. Accepted formats: ${ACCEPTED_FILE_TYPES}.`)
        } else {
          setDropzoneError('Some files could not be added. Please try different files.')
          pushToast('error', 'Some files could not be added. Please choose different images.')
        }
      } else {
        setDropzoneError('')
      }

      if (acceptedFiles.length === 0) {
        return
      }

      const preparedItems = await Promise.all(
        acceptedFiles.map(async (file) => {
          const preview = await createOptimizedPreview(file)
          return {
            id: `${file.name}-${file.lastModified}-${Math.random().toString(16).slice(2)}`,
            file,
            previewUrl: preview.previewUrl,
            name: file.name,
            sizeLabel: formatFileSize(file.size),
            type: file.type || 'image/*',
            width: preview.width,
            height: preview.height,
          }
        }),
      )

      if (uploadMode === 'single') {
        replaceUploadedItems(preparedItems.slice(0, 1))
      } else {
        appendUploadedItems(preparedItems)
      }

      pushToast('success', `${preparedItems.length} image${preparedItems.length > 1 ? 's' : ''} ready for analysis.`)
    },
    [appendUploadedItems, pushToast, replaceUploadedItems, uploadMode],
  )

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: { 'image/*': [] },
    maxSize: MAX_FILE_SIZE,
    multiple: uploadMode === 'batch',
    noClick: true,
  })

  function removeUploadedItem(itemId) {
    setUploadedItems((previous) => {
      const item = previous.find((entry) => entry.id === itemId)
      if (item) {
        URL.revokeObjectURL(item.previewUrl)
      }
      return previous.filter((entry) => entry.id !== itemId)
    })
  }

  async function applyBackendConfig() {
    await runConfigRequest(() =>
      updateConfig({
        default_standard: selectedStandard,
        default_weighting_method: selectedWeightingMethod,
        default_adaptive_scoring: adaptiveScoring,
      }),
    )
  }

  async function runAssessment() {
    if (uploadedItems.length === 0) {
      setFormError('Select at least one image file first.')
      pushToast('error', 'Select at least one image before processing.')
      return
    }

    setLastAttemptMode(uploadMode)
    setFormError('')
    setSingleResult(null)
    setBatchResults([])
    setBatchSummary([])
    setExportMessage('')
    setUploadProgress(8)
    setProcessStageText('Preparing request')

    try {
      setUploadProgress(18)
      setProcessStageText('Applying backend configuration')
      await applyBackendConfig()

      if (uploadMode === 'single') {
        setUploadProgress(32)
        setProcessStageText('Uploading image')
        const payload = await runPredictRequest(() =>
          assessImage(uploadedItems[0].file, {
            adaptiveScoring,
            timeoutMs: 25000,
            retryCount: 2,
          }),
        )
        setSingleResult(payload?.data?.assessment ?? null)
        setResultsTab('overview')
        pushToast('success', 'Image processed successfully.')
      } else {
        setUploadProgress(32)
        setProcessStageText('Uploading batch')
        const payload = await runPredictRequest(() =>
          assessBatch(
            uploadedItems.map((item) => item.file),
            {
              adaptiveScoring,
              timeoutMs: 40000,
              retryCount: 2,
            },
          ),
        )

        const records = payload?.data?.results ?? []
        setBatchResults(records)
        setBatchSummary(summarizeBatch(records))
        setResultsTab('overview')
        pushToast('success', `Batch processed successfully (${records.length} results).`)
      }
    } catch (error) {
      void error
      pushToast('error', 'Analysis failed. See guidance below and retry.')
    } finally {
      setUploadProgress(100)
      setProcessStageText('Completed')
    }
  }

  async function handleProcess(event) {
    event.preventDefault()
    await runAssessment()
  }

  const sourceInfo = useMemo(() => normalizeSourceInfo(singleResult?.turbidity_source), [singleResult])
  const severityConfig = useMemo(
    () => getSeverityConfig(singleResult?.turbidity_category),
    [singleResult],
  )
  const sourceChartData = useMemo(() => buildSourceChartData(sourceInfo), [sourceInfo])
  const featureChartData = useMemo(
    () => buildFeatureChartData(singleResult?.feature_contributions),
    [singleResult],
  )
  const probabilityData = useMemo(
    () => buildProbabilityData(singleResult?.probabilistic_classification),
    [singleResult],
  )

  const hasSingleResult = uploadMode === 'single' && Boolean(singleResult)
  const hasBatchResults = uploadMode === 'batch' && batchResults.length > 0
  const hasAnyResult = hasSingleResult || hasBatchResults
  const primaryItem = uploadedItems[0] || null

  const visibilityScore = Number(singleResult?.visibility_score ?? 0)
  const visibilityPercent = clamp(visibilityScore * 100, 0, 100)
  const confidencePercent = clamp(Number(singleResult?.confidence_numeric ?? 0) * 100, 0, 100)

  const equivalentMetricsRows = useMemo(() => {
    if (!singleResult?.equivalent_metrics) {
      return []
    }

    const metrics = singleResult.equivalent_metrics
    return [
      {
        label: 'Estimated NTU',
        value: metrics.estimated_ntu ?? 'N/A',
        range: Array.isArray(metrics.ntu_range) ? `${metrics.ntu_range[0]} to ${metrics.ntu_range[1]}` : 'N/A',
      },
      {
        label: 'Estimated Secchi Depth (m)',
        value: metrics.estimated_secchi_depth_m ?? 'N/A',
        range: Array.isArray(metrics.secchi_depth_range)
          ? `${metrics.secchi_depth_range[0]} to ${metrics.secchi_depth_range[1]}`
          : 'N/A',
      },
      {
        label: 'Carlson TSI',
        value: metrics.tsi_applicable ? metrics.estimated_carlson_tsi ?? 'N/A' : 'N/A',
        range: Array.isArray(metrics.tsi_range) ? `${metrics.tsi_range[0]} to ${metrics.tsi_range[1]}` : metrics.tsi_note || 'Not applicable',
      },
      {
        label: 'Classification Standard',
        value: singleResult.standard_used ?? 'N/A',
        range: 'Selected standard used by backend classifier',
      },
    ]
  }, [singleResult])

  const overviewSummaryCards = useMemo(() => {
    if (!hasSingleResult) {
      return []
    }

    return [
      ['Category', singleResult.turbidity_category],
      ['Visibility Score', Number(singleResult.visibility_score ?? 0).toFixed(3)],
      ['Confidence', `${singleResult.confidence || 'N/A'} (${Number(singleResult.confidence_numeric ?? 0).toFixed(2)})`],
      ['Source', titleCase(sourceInfo.primary_source)],
      ['Source Confidence', Number(sourceInfo.confidence ?? 0).toFixed(2)],
      ['YOLO Confidence', Number(singleResult.yolo_confidence ?? 0).toFixed(3)],
    ]
  }, [hasSingleResult, singleResult, sourceInfo])

  const canProcess = status.initialized && uploadedItems.length > 0 && !isConfigLoading

  async function handleDownloadAnnotatedImage() {
    if (!primaryItem || !singleResult) {
      return
    }

    try {
      const annotatedDataUrl = await drawAnnotatedImage(primaryItem.previewUrl, singleResult)
      const link = document.createElement('a')
      link.href = annotatedDataUrl
      link.download = `annotated-${primaryItem.name.replace(/\s+/g, '-')}.png`
      document.body.appendChild(link)
      link.click()
      link.remove()
      setExportMessage('Annotated image downloaded.')
    } catch {
      setExportMessage('Failed to generate annotated image.')
    }
  }

  function handleExportJson() {
    if (!hasAnyResult) {
      return
    }

    const payload = {
      exported_at: new Date().toISOString(),
      mode: uploadMode,
      configuration: {
        standard: selectedStandard,
        weighting: selectedWeightingMethod,
        adaptive_scoring: adaptiveScoring,
      },
      result: hasSingleResult ? singleResult : batchResults,
    }

    downloadBlob('turbidity-result.json', 'application/json', JSON.stringify(payload, null, 2))
    setExportMessage('Result JSON exported.')
  }

  async function handleCopyResult() {
    if (!hasAnyResult) {
      return
    }

    const payload = hasSingleResult ? singleResult : batchResults

    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2))
      setExportMessage('Results copied to clipboard.')
    } catch {
      setExportMessage('Clipboard copy failed. Browser permissions may block this action.')
    }
  }

  function handleSaveComparison() {
    if (!singleResult) {
      return
    }

    const nextSnapshot = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      timestamp: new Date().toLocaleString(),
      category: singleResult.turbidity_category,
      score: Number(singleResult.visibility_score ?? 0),
      source: normalizeSourceInfo(singleResult.turbidity_source).primary_source,
      confidence: Number(singleResult.confidence_numeric ?? 0),
      ntu: singleResult.equivalent_metrics?.estimated_ntu ?? 'N/A',
    }

    setComparisonSnapshots((previous) => [...previous.slice(-4), nextSnapshot])
    setExportMessage('Snapshot saved for comparison.')
  }

  function handleResetFlow() {
    setSingleResult(null)
    setBatchResults([])
    setBatchSummary([])
    setFormError('')
    setDropzoneError('')
    setResultsTab('overview')
    setImageZoom(1)
    setComparisonSnapshots([])
    setExportMessage('Flow reset. Upload another image to continue.')
    replaceUploadedItems([])
  }

  function handleImageZoomToggle() {
    setImageZoom((previous) => (previous > 1 ? 1 : 2))
  }

  const previousSnapshot = comparisonSnapshots.length > 0 ? comparisonSnapshots[comparisonSnapshots.length - 1] : null
  const currentComparison = hasSingleResult
    ? {
        category: singleResult.turbidity_category,
        score: Number(singleResult.visibility_score ?? 0),
        source: sourceInfo.primary_source,
        confidence: Number(singleResult.confidence_numeric ?? 0),
        ntu: singleResult.equivalent_metrics?.estimated_ntu ?? 'N/A',
      }
    : null

  const noDiskDetected = hasSingleResult && singleResult.disk_detected === false
  const errorInfo = getFriendlyErrorInfo(isPredictError ? predictError : formError, noDiskDetected ? singleResult : null)
  const canRetry = uploadedItems.length > 0 && !isPredictLoading && lastAttemptMode === uploadMode

  return (
    <main
      className="mx-auto grid min-h-screen w-full max-w-6xl gap-6 px-4 py-6 font-sans md:px-8 md:py-8"
      aria-busy={isPredictLoading || isStatusLoading || isConfigLoading}
    >
      <div className="pointer-events-none fixed right-4 top-4 z-50 grid w-[min(24rem,92vw)] gap-2" aria-live="polite" aria-atomic="false">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`pointer-events-auto rounded-xl border px-3 py-2 text-sm shadow transition-all ${
              toast.type === 'error'
                ? 'border-red-200 bg-red-50 text-red-700'
                : toast.type === 'warning'
                  ? 'border-amber-200 bg-amber-50 text-amber-800'
                  : 'border-emerald-200 bg-emerald-50 text-emerald-700'
            }`}
            role="status"
          >
            {toast.message}
          </div>
        ))}
      </div>

      <header className="rounded-3xl border border-sage-200 bg-gradient-to-br from-algae-700 via-algae-600 to-forest-500 p-6 text-mint-50 shadow-primary md:p-8">
        <p className="m-0 text-xs uppercase tracking-[0.2em] text-algae-100">Secchi Disk Detection</p>
        <h1 className="mt-3 text-3xl font-bold tracking-tight text-white md:text-4xl">Turbidity Classifier</h1>
        <p className="mt-2 max-w-3xl text-sm text-algae-50 md:text-base">
          Upload images, tune processing options, and run source-aware turbidity assessment.
        </p>
      </header>

      <section className="grid grid-cols-1 gap-5 xl:grid-cols-[1.6fr_1fr]">
        <article className="rounded-2xl border border-sage-200 bg-white p-5 shadow">
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => handleUploadModeChange('single')}
              aria-pressed={uploadMode === 'single'}
              className={`rounded-lg px-3 py-1.5 text-sm font-semibold transition ${
                uploadMode === 'single'
                  ? 'bg-algae-600 text-white'
                  : 'bg-mint-100 text-moss-600 hover:bg-sage-100'
              } focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-algae-600`}
            >
              Single Image
            </button>
            <button
              type="button"
              onClick={() => handleUploadModeChange('batch')}
              aria-pressed={uploadMode === 'batch'}
              className={`rounded-lg px-3 py-1.5 text-sm font-semibold transition ${
                uploadMode === 'batch'
                  ? 'bg-algae-600 text-white'
                  : 'bg-mint-100 text-moss-600 hover:bg-sage-100'
              } focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-algae-600`}
            >
              Batch Upload
            </button>
          </div>

          <div
            {...getRootProps({
              role: 'button',
              tabIndex: 0,
              'aria-label': 'Upload image files',
              onKeyDown: (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  open()
                }
              },
            })}
            className={`rounded-2xl border-2 border-dashed p-6 text-center transition ${
              isDragActive
                ? 'border-algae-500 bg-algae-50'
                : 'border-sage-300 bg-mint-100/70 hover:border-algae-400 hover:bg-algae-50'
            }`}
          >
            <input {...getInputProps()} />
            <p className="text-sm font-medium text-moss-700">
              Drag and drop image{uploadMode === 'batch' ? 's' : ''} here
            </p>
            <p className="mt-1 text-xs text-moss-500">
              {uploadMode === 'batch'
                ? 'Upload multiple JPG/PNG/WebP files, max 10 MB each'
                : 'Upload one JPG/PNG/WebP file, max 10 MB'}
            </p>
            <button
              type="button"
              onClick={open}
              className="mt-4 rounded-xl border border-sage-300 bg-white px-4 py-2 text-sm font-semibold text-moss-700 transition hover:border-algae-400 hover:text-algae-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-algae-600"
            >
              Click to upload
            </button>
          </div>

          {dropzoneError && <p className="mt-3 text-sm font-semibold text-red-600">{dropzoneError}</p>}

          <div className="mt-5">
            <div className="mb-3 flex items-center justify-between gap-3">
              <h3 className="text-base font-semibold text-moss-900">Selected Files</h3>
              {uploadedItems.length > 0 && (
                <span className="rounded-full bg-forest-100 px-2.5 py-1 text-xs font-semibold text-forest-700">
                  {uploadedItems.length}
                </span>
              )}
            </div>

            {uploadedItems.length === 0 ? (
              <p className="text-sm text-moss-500">No image selected yet.</p>
            ) : (
              <div className="grid gap-3 sm:grid-cols-2">
                {uploadedItems.map((item) => (
                  <div key={item.id} className="rounded-xl border border-sage-200 bg-white p-3 shadow-sm">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="max-w-[180px] truncate text-sm font-semibold text-moss-900" title={item.name}>
                          {item.name}
                        </p>
                        <p className="text-xs text-moss-500">{item.sizeLabel} · {item.type}</p>
                        <p className="text-xs text-moss-500">
                          {item.width && item.height ? `${item.width} x ${item.height}` : 'Dimensions unavailable'}
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => removeUploadedItem(item.id)}
                        className="rounded-lg border border-sage-300 px-2 py-1 text-xs font-semibold text-moss-600 transition hover:border-red-300 hover:text-red-600"
                      >
                        Remove
                      </button>
                    </div>

                    <img
                      src={item.previewUrl}
                      alt={item.name}
                      className="mt-3 h-32 w-full rounded-lg border border-sage-200 object-cover"
                    />
                  </div>
                ))}
              </div>
            )}
          </div>

          <form className="mt-5" onSubmit={handleProcess}>
            <button
              type="submit"
              disabled={!canProcess || isPredictLoading}
              className="w-full rounded-xl bg-gradient-to-r from-algae-600 to-algae-500 px-4 py-3 text-sm font-bold text-white shadow-primary transition hover:from-algae-700 hover:to-algae-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-algae-600 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isPredictLoading
                ? uploadMode === 'batch'
                  ? 'Processing Batch...'
                  : 'Processing Image...'
                : uploadMode === 'batch'
                  ? 'Process Batch'
                  : 'Process Image'}
            </button>
          </form>

          {(isPredictLoading || isConfigLoading || isStatusLoading) && (
            <div className="mt-3 rounded-xl border border-sage-200 bg-mint-100 p-3" role="status" aria-live="polite">
              <div className="mb-2 flex items-center justify-between text-xs font-medium uppercase tracking-[0.08em] text-moss-600">
                <span>{processStageText}</span>
                <span>{Math.round(uploadProgress)}%</span>
              </div>
              <div className="h-2.5 rounded-full bg-sage-200">
                <div
                  className="h-2.5 rounded-full bg-gradient-to-r from-algae-600 to-forest-500 transition-[width] duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="mt-2 text-xs text-moss-500">Active requests: {predictActiveRequests + statusActiveRequests}</p>
              <div className="mt-2 grid gap-2 sm:grid-cols-3">
                <LoadingSkeleton className="h-8 w-full" />
                <LoadingSkeleton className="h-8 w-full" />
                <LoadingSkeleton className="h-8 w-full" />
              </div>
            </div>
          )}

          {formError && <p className="mt-3 text-sm font-semibold text-red-600">{formError}</p>}

          {(isPredictError || noDiskDetected) && (
            <div className="mt-3 rounded-xl border border-red-200 bg-red-50 p-3 text-red-800" role="alert">
              <p className="text-sm font-bold">{errorInfo.title}</p>
              <p className="mt-1 text-sm">{errorInfo.description}</p>
              <ul className="mt-2 list-inside list-disc text-xs">
                {errorInfo.suggestions.map((suggestion) => (
                  <li key={suggestion}>{suggestion}</li>
                ))}
              </ul>
              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => {
                    void runAssessment()
                  }}
                  disabled={!canRetry}
                  className="rounded-lg border border-red-300 bg-white px-3 py-1.5 text-xs font-semibold text-red-700 hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Retry
                </button>
                <button
                  type="button"
                  onClick={handleResetFlow}
                  className="rounded-lg border border-red-300 bg-white px-3 py-1.5 text-xs font-semibold text-red-700 hover:bg-red-100"
                >
                  Upload Different Image
                </button>
              </div>
            </div>
          )}

          {isConfigError && <p className="mt-3 text-sm font-semibold text-red-600">{configError}</p>}
          {isStatusError && <p className="mt-3 text-sm font-semibold text-red-600">{statusError}</p>}
          {!status.initialized && (
            <p className="mt-3 text-sm font-medium text-amber-700">Initialize the system before running predictions.</p>
          )}
        </article>

        <article className="rounded-2xl border border-sage-200 bg-white p-5 shadow">
          <div className="mb-4 flex flex-col items-start justify-between gap-3 sm:flex-row sm:items-center">
            <h2 className="text-lg font-semibold text-moss-900">Configuration Panel</h2>
            <button
              type="button"
              onClick={refreshStatus}
              disabled={isStatusLoading}
              className="rounded-xl bg-algae-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-algae-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isStatusLoading ? 'Refreshing...' : 'Sync Backend'}
            </button>
          </div>

          <div
            className={`mb-3 inline-flex rounded-full px-3 py-1 text-sm font-bold ${
              status.initialized
                ? 'bg-forest-100 text-forest-700'
                : 'bg-amber-100 text-amber-800'
            }`}
          >
            {status.initialized ? 'Backend Ready' : 'Backend Not Ready'}
          </div>

          {status.model_path && <p className="mt-2 text-sm text-moss-600">Current model: {status.model_path}</p>}

          <hr className="my-5 border-sage-200" />

          <div className="grid gap-3">
            <label className="grid gap-1.5 text-sm font-semibold text-moss-700">
              Turbidity Standard
              <select
                value={selectedStandard}
                onChange={(event) => setSelectedStandard(event.target.value)}
                className="rounded-xl border border-sage-200 bg-mint-100 px-3 py-2 text-sm text-moss-900"
                title="Classification standard used by the backend"
              >
                <option value="auto">Auto (source-aware)</option>
                <option value="carlson">Carlson</option>
                <option value="sediment">Sediment</option>
                <option value="epa">EPA</option>
                <option value="marine">Marine</option>
                <option value="freshwater">Freshwater</option>
              </select>
            </label>

            <label className="grid gap-1.5 text-sm font-semibold text-moss-700">
              Weighting Method
              <select
                value={selectedWeightingMethod}
                onChange={(event) => setSelectedWeightingMethod(event.target.value)}
                className="rounded-xl border border-sage-200 bg-mint-100 px-3 py-2 text-sm text-moss-900"
                title="Feature weighting strategy for visibility scoring"
              >
                <option value="balanced">Balanced</option>
                <option value="physics">Physics</option>
                <option value="edge_focused">Edge-focused</option>
              </select>
            </label>

            <label className="inline-flex items-center gap-2 text-sm text-moss-600">
              <input
                type="checkbox"
                checked={adaptiveScoring}
                onChange={(event) => setAdaptiveScoring(event.target.checked)}
                className="h-4 w-4 rounded border-sage-300 text-algae-600 focus:ring-algae-600"
              />
              Adaptive scoring
            </label>

            <label className="inline-flex items-center gap-2 text-sm text-moss-600">
              <input
                type="checkbox"
                checked={showDetailedOutput}
                onChange={(event) => setShowDetailedOutput(event.target.checked)}
                className="h-4 w-4 rounded border-sage-300 text-algae-600 focus:ring-algae-600"
              />
              Show detailed metrics
            </label>

            <label className="inline-flex items-center gap-2 text-sm text-moss-600">
              <input
                type="checkbox"
                checked={showSourceBreakdown}
                onChange={(event) => setShowSourceBreakdown(event.target.checked)}
                className="h-4 w-4 rounded border-sage-300 text-algae-600 focus:ring-algae-600"
              />
              Show source breakdown
            </label>
          </div>
        </article>
      </section>

      <section className="min-h-44 rounded-2xl border border-sage-200 bg-white p-5 shadow print:border-0 print:shadow-none">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-semibold text-moss-900">Prediction Output</h2>
          <div className="inline-flex rounded-xl border border-sage-200 bg-mint-100 p-1">
            <button
              type="button"
              onClick={() => setResultsTab('overview')}
              aria-pressed={resultsTab === 'overview'}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${
                resultsTab === 'overview' ? 'bg-algae-600 text-white' : 'text-moss-700'
              }`}
            >
              Overview
            </button>
            <button
              type="button"
              onClick={() => setResultsTab('details')}
              aria-pressed={resultsTab === 'details'}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${
                resultsTab === 'details' ? 'bg-algae-600 text-white' : 'text-moss-700'
              }`}
            >
              Detailed
            </button>
            <button
              type="button"
              onClick={() => setResultsTab('actions')}
              aria-pressed={resultsTab === 'actions'}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${
                resultsTab === 'actions' ? 'bg-algae-600 text-white' : 'text-moss-700'
              }`}
            >
              Actions
            </button>
          </div>
        </div>

        {!hasAnyResult && <p className="text-sm text-moss-500">No prediction yet. Upload images and process.</p>}

        {hasBatchResults && (
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-xl border border-sage-200 bg-mint-100 p-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.08em] text-moss-600">Category Summary</h3>
              <div className="mt-3 grid gap-2">
                {batchSummary.map(([category, count]) => (
                  <div key={category} className="flex items-center justify-between rounded-lg bg-white px-3 py-2">
                    <span className="text-sm font-medium text-moss-700">{category}</span>
                    <span className="rounded-full bg-forest-100 px-2.5 py-1 text-xs font-semibold text-forest-700">{count}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-xl border border-sage-200 bg-mint-100 p-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.08em] text-moss-600">Batch Results</h3>
              <div className="mt-3 max-h-64 space-y-2 overflow-auto pr-1">
                {batchResults.map((entry, index) => {
                  const entrySource = normalizeSourceInfo(entry.turbidity_source || entry.source)
                  return (
                    <div key={`${entry.image_path}-${index}`} className="rounded-lg bg-white px-3 py-2">
                      <p className="text-xs text-moss-500">{entry.image_path}</p>
                      <p className="text-sm font-semibold text-moss-900">{entry.turbidity_category}</p>
                      {showSourceBreakdown && (
                        <p className="text-xs text-moss-600">Source: {titleCase(entrySource.primary_source)}</p>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}

        {hasSingleResult && resultsTab === 'overview' && (
          <div className="grid gap-5">
            <section className="rounded-2xl border border-sage-200 bg-gradient-to-br from-algae-600 to-forest-500 p-5 text-white md:p-6">
              <p className="text-xs uppercase tracking-[0.14em] text-algae-100">Primary Classification</p>
              <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                <h3 className="text-2xl font-bold md:text-3xl">{singleResult.turbidity_category || 'Unknown Category'}</h3>
                <span className={`rounded-full px-3 py-1 text-xs font-bold ${severityConfig.badgeClass}`}>
                  {severityConfig.label}
                </span>
              </div>

              <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <div className="rounded-xl bg-white/15 p-3 backdrop-blur-sm">
                  <p className="text-xs uppercase tracking-[0.08em] text-algae-100">Visibility Score</p>
                  <p className="mt-1 text-2xl font-bold">{Number(singleResult.visibility_score ?? 0).toFixed(3)}</p>
                </div>
                <div className="rounded-xl bg-white/15 p-3 backdrop-blur-sm">
                  <p className="text-xs uppercase tracking-[0.08em] text-algae-100">Confidence</p>
                  <p className="mt-1 text-2xl font-bold">{singleResult.confidence || 'N/A'}</p>
                </div>
                <div className="rounded-xl bg-white/15 p-3 backdrop-blur-sm">
                  <p className="text-xs uppercase tracking-[0.08em] text-algae-100">Source</p>
                  <p className="mt-1 text-2xl font-bold">{titleCase(sourceInfo.primary_source)}</p>
                </div>
                <div className="rounded-xl bg-white/15 p-3 backdrop-blur-sm">
                  <p className="text-xs uppercase tracking-[0.08em] text-algae-100">Standard</p>
                  <p className="mt-1 text-2xl font-bold">{String(singleResult.standard_used || 'N/A').toUpperCase()}</p>
                </div>
              </div>

              <div className="mt-5 grid gap-4 md:grid-cols-2">
                <div>
                  <div className="mb-1 flex items-center justify-between text-xs">
                    <span>Visibility indicator</span>
                    <span>{visibilityPercent.toFixed(1)}%</span>
                  </div>
                  <div className="h-2.5 rounded-full bg-white/20">
                    <div
                      className={`h-2.5 rounded-full ${severityConfig.barClass}`}
                      style={{ width: `${visibilityPercent}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="mb-1 flex items-center justify-between text-xs">
                    <span>Classification confidence</span>
                    <span>{confidencePercent.toFixed(1)}%</span>
                  </div>
                  <div className="h-2.5 rounded-full bg-white/20">
                    <div className="h-2.5 rounded-full bg-white" style={{ width: `${confidencePercent}%` }} />
                  </div>
                </div>
              </div>
            </section>

            <section className="grid gap-4 lg:grid-cols-2">
              <div className="rounded-2xl border border-sage-200 bg-mint-100 p-4">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                  <h3 className="text-sm font-semibold uppercase tracking-[0.08em] text-moss-700">Annotated Image Viewer</h3>
                  <div className="flex items-center gap-2">
                    <span className="rounded-full bg-white px-2.5 py-1 text-xs font-semibold text-moss-600">
                      Click image to {imageZoom > 1 ? 'reset' : 'zoom'}
                    </span>
                    <button
                      type="button"
                      onClick={handleDownloadAnnotatedImage}
                      className="rounded-lg border border-sage-300 bg-white px-3 py-1.5 text-xs font-semibold text-moss-700 hover:border-algae-400 hover:text-algae-700"
                    >
                      Download Image
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-3">
                  <div className="rounded-xl border border-sage-200 bg-white p-2">
                    <p className="mb-2 text-xs uppercase tracking-[0.08em] text-moss-500">Original</p>
                    {primaryItem ? (
                      <button
                        type="button"
                        onClick={handleImageZoomToggle}
                        className="block w-full overflow-hidden rounded-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-algae-600"
                        aria-label="Toggle zoom for original image"
                      >
                        <img
                          src={primaryItem.previewUrl}
                          alt={primaryItem.name}
                          className="h-56 w-full rounded-lg object-cover transition-transform duration-200"
                          style={{ transform: `scale(${debouncedImageZoom})`, transformOrigin: 'center center' }}
                        />
                      </button>
                    ) : (
                      <div className="flex h-56 items-center justify-center text-sm text-moss-500">No image selected</div>
                    )}
                  </div>

                  <div className="rounded-xl border border-sage-200 bg-white p-2">
                    <p className="mb-2 text-xs uppercase tracking-[0.08em] text-moss-500">Annotated</p>
                    {primaryItem ? (
                      <button
                        type="button"
                        onClick={handleImageZoomToggle}
                        className="block w-full overflow-hidden rounded-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-algae-600"
                        aria-label="Toggle zoom for annotated image"
                      >
                        <div className="relative overflow-hidden rounded-lg">
                          <img
                            src={primaryItem.previewUrl}
                            alt={`${primaryItem.name} annotation`}
                            className="h-56 w-full object-cover transition-transform duration-200"
                            style={{ transform: `scale(${debouncedImageZoom})`, transformOrigin: 'center center' }}
                          />
                          {Array.isArray(singleResult.bbox) && primaryItem.width && primaryItem.height && (
                            <div
                              className="pointer-events-none absolute border-2 border-algae-600"
                              style={{
                                left: `${(Number(singleResult.bbox[0]) / primaryItem.width) * 100}%`,
                                top: `${(Number(singleResult.bbox[1]) / primaryItem.height) * 100}%`,
                                width: `${((Number(singleResult.bbox[2]) - Number(singleResult.bbox[0])) / primaryItem.width) * 100}%`,
                                height: `${((Number(singleResult.bbox[3]) - Number(singleResult.bbox[1])) / primaryItem.height) * 100}%`,
                              }}
                            >
                              <span className="absolute -top-6 left-0 rounded bg-algae-600 px-2 py-0.5 text-[10px] font-semibold text-white">
                                {singleResult.turbidity_category} | {Number(singleResult.visibility_score ?? 0).toFixed(3)}
                              </span>
                            </div>
                          )}
                        </div>
                      </button>
                    ) : (
                      <div className="flex h-56 items-center justify-center text-sm text-moss-500">No image selected</div>
                    )}
                  </div>
                </div>
              </div>

              <div className="grid gap-4">
                <div className="rounded-2xl border border-sage-200 bg-mint-100 p-4">
                  <h3 className="text-sm font-semibold uppercase tracking-[0.08em] text-moss-700">Source Detection Information</h3>
                  <p className="mt-1 text-sm text-moss-600">{sourceDescription(sourceInfo.primary_source)}</p>
                  <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                    <div className="rounded-lg bg-white px-3 py-2">
                      <p className="text-xs text-moss-500">Primary source</p>
                      <p className="font-semibold text-moss-900">{titleCase(sourceInfo.primary_source)}</p>
                    </div>
                    <div className="rounded-lg bg-white px-3 py-2">
                      <p className="text-xs text-moss-500">Source confidence</p>
                      <p className="font-semibold text-moss-900">{toPercent(sourceInfo.confidence || 0, 2)}</p>
                    </div>
                  </div>

                  <div className="mt-3 h-48 rounded-lg bg-white p-2">
                    <Suspense fallback={<ChartFallback heightClass="h-44" />}>
                      <SourcePieChart data={sourceChartData} />
                    </Suspense>
                  </div>
                </div>

                <div className="rounded-2xl border border-sage-200 bg-mint-100 p-4">
                  <h3 className="text-sm font-semibold uppercase tracking-[0.08em] text-moss-700">Equivalent Metrics</h3>
                  <div className="mt-3 overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="text-xs uppercase tracking-[0.08em] text-moss-500">
                          <th className="pb-2">Metric</th>
                          <th className="pb-2">Value</th>
                          <th className="pb-2">Range / Note</th>
                        </tr>
                      </thead>
                      <tbody>
                        {equivalentMetricsRows.map((row) => (
                          <tr key={row.label} className="border-t border-sage-200">
                            <td className="py-2 pr-3 font-semibold text-moss-800">{row.label}</td>
                            <td className="py-2 pr-3 text-moss-700">{String(row.value)}</td>
                            <td className="py-2 text-moss-600">{row.range}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="mt-2 text-xs text-moss-500">
                    Ranges indicate uncertainty bands estimated by the classifier and source-aware standard selection.
                  </p>
                </div>
              </div>
            </section>

            <section className="rounded-2xl border border-sage-200 bg-mint-100 p-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.08em] text-moss-700">Quick Summary Cards</h3>
              <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {(showDetailedOutput ? overviewSummaryCards : overviewSummaryCards.slice(0, 3)).map(([label, value]) => (
                  <div key={label} className="rounded-xl border border-sage-200 bg-white p-3 shadow-sm">
                    <p className="m-0 text-[11px] uppercase tracking-[0.08em] text-moss-500">{label}</p>
                    <p className="mt-1 text-base font-bold text-moss-900">{String(value)}</p>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}

        {hasSingleResult && resultsTab === 'details' && (
          <div className="grid gap-4">
            <section className="rounded-2xl border border-sage-200 bg-mint-100 p-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.08em] text-moss-700">Feature Contributions</h3>
              <p className="mt-1 text-sm text-moss-600">Top weighted normalized features driving the final visibility score.</p>
              <div className="mt-3 h-72 rounded-lg bg-white p-2">
                <Suspense fallback={<ChartFallback heightClass="h-64" />}>
                  <FeatureBarChart data={featureChartData} />
                </Suspense>
              </div>
            </section>

            <section className="rounded-2xl border border-sage-200 bg-mint-100 p-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.08em] text-moss-700">Detailed Information</h3>

              <div className="mt-3 grid gap-3">
                <details className="rounded-xl border border-sage-200 bg-white p-3" open>
                  <summary className="cursor-pointer text-sm font-semibold text-moss-800">Category Profile and Explanation</summary>
                  <div className="mt-2 text-sm text-moss-700">
                    <p>Description: {singleResult.category_info?.description || 'N/A'}</p>
                    <p>Visibility: {singleResult.category_info?.visibility || 'N/A'}</p>
                    <p>Typical causes: {singleResult.category_info?.typical_causes || 'N/A'}</p>
                  </div>
                </details>

                <details className="rounded-xl border border-sage-200 bg-white p-3">
                  <summary className="cursor-pointer text-sm font-semibold text-moss-800">Management and Ecological Impact</summary>
                  <div className="mt-2 text-sm text-moss-700">
                    <p>Management recommendation: {singleResult.category_info?.management_action || 'N/A'}</p>
                    <p>Ecological impact: {singleResult.category_info?.ecological_impact || 'N/A'}</p>
                    <p>Trophic state: {singleResult.category_info?.trophic_state || 'N/A'}</p>
                  </div>
                </details>

                <details className="rounded-xl border border-sage-200 bg-white p-3">
                  <summary className="cursor-pointer text-sm font-semibold text-moss-800">Probabilistic Classification Breakdown</summary>
                  <div className="mt-2 h-56">
                    <Suspense fallback={<ChartFallback heightClass="h-52" />}>
                      <ProbabilityBarChart data={probabilityData} />
                    </Suspense>
                  </div>
                </details>

                <details className="rounded-xl border border-sage-200 bg-white p-3">
                  <summary className="cursor-pointer text-sm font-semibold text-moss-800">Raw and Normalized Features</summary>
                  <div className="mt-2 grid gap-2 sm:grid-cols-2">
                    <div>
                      <p className="mb-1 text-xs uppercase tracking-[0.08em] text-moss-500">Raw Features</p>
                      <pre className="max-h-56 overflow-auto rounded-lg bg-mint-100 p-2 text-xs text-moss-700">
                        {JSON.stringify(singleResult.features || {}, null, 2)}
                      </pre>
                    </div>
                    <div>
                      <p className="mb-1 text-xs uppercase tracking-[0.08em] text-moss-500">Normalized Features</p>
                      <pre className="max-h-56 overflow-auto rounded-lg bg-mint-100 p-2 text-xs text-moss-700">
                        {JSON.stringify(singleResult.normalized_features || {}, null, 2)}
                      </pre>
                    </div>
                  </div>
                </details>

                <details className="rounded-xl border border-sage-200 bg-white p-3">
                  <summary className="cursor-pointer text-sm font-semibold text-moss-800">Weighting Method Comparison</summary>
                  <pre className="mt-2 max-h-56 overflow-auto rounded-lg bg-mint-100 p-2 text-xs text-moss-700">
                    {JSON.stringify(singleResult.method_comparison || {}, null, 2)}
                  </pre>
                </details>
              </div>
            </section>
          </div>
        )}

        {hasAnyResult && resultsTab === 'actions' && (
          <div className="grid gap-4">
            <section className="rounded-2xl border border-sage-200 bg-mint-100 p-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.08em] text-moss-700">Result Actions</h3>
              <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
                <button
                  type="button"
                  onClick={handleExportJson}
                  className="rounded-lg border border-sage-300 bg-white px-3 py-2 text-sm font-semibold text-moss-700 hover:border-algae-400 hover:text-algae-700"
                >
                  Export JSON
                </button>
                <button
                  type="button"
                  onClick={() => window.print()}
                  className="rounded-lg border border-sage-300 bg-white px-3 py-2 text-sm font-semibold text-moss-700 hover:border-algae-400 hover:text-algae-700"
                >
                  Export Report (Print/PDF)
                </button>
                <button
                  type="button"
                  onClick={handleCopyResult}
                  className="rounded-lg border border-sage-300 bg-white px-3 py-2 text-sm font-semibold text-moss-700 hover:border-algae-400 hover:text-algae-700"
                >
                  Copy to Clipboard
                </button>
                <button
                  type="button"
                  onClick={handleResetFlow}
                  className="rounded-lg border border-sage-300 bg-white px-3 py-2 text-sm font-semibold text-moss-700 hover:border-amber-400 hover:text-amber-700"
                >
                  Analyze Another Image
                </button>
                <button
                  type="button"
                  onClick={handleSaveComparison}
                  disabled={!hasSingleResult}
                  className="rounded-lg border border-sage-300 bg-white px-3 py-2 text-sm font-semibold text-moss-700 hover:border-algae-400 hover:text-algae-700 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Compare Snapshot
                </button>
              </div>

              {exportMessage && <p className="mt-3 text-sm text-moss-600">{exportMessage}</p>}
            </section>

            {hasSingleResult && (
              <section className="rounded-2xl border border-sage-200 bg-mint-100 p-4">
                <h3 className="text-sm font-semibold uppercase tracking-[0.08em] text-moss-700">Comparison View</h3>
                {!previousSnapshot && (
                  <p className="mt-2 text-sm text-moss-600">
                    Save a snapshot from one run, then process another image to compare side-by-side.
                  </p>
                )}

                {previousSnapshot && currentComparison && (
                  <div className="mt-3 overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="text-xs uppercase tracking-[0.08em] text-moss-500">
                          <th className="pb-2">Metric</th>
                          <th className="pb-2">Previous ({previousSnapshot.timestamp})</th>
                          <th className="pb-2">Current</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-t border-sage-200">
                          <td className="py-2 font-semibold text-moss-800">Category</td>
                          <td className="py-2 text-moss-700">{previousSnapshot.category}</td>
                          <td className="py-2 text-moss-700">{currentComparison.category}</td>
                        </tr>
                        <tr className="border-t border-sage-200">
                          <td className="py-2 font-semibold text-moss-800">Visibility Score</td>
                          <td className="py-2 text-moss-700">{previousSnapshot.score.toFixed(3)}</td>
                          <td className="py-2 text-moss-700">{currentComparison.score.toFixed(3)}</td>
                        </tr>
                        <tr className="border-t border-sage-200">
                          <td className="py-2 font-semibold text-moss-800">Source</td>
                          <td className="py-2 text-moss-700">{titleCase(previousSnapshot.source)}</td>
                          <td className="py-2 text-moss-700">{titleCase(currentComparison.source)}</td>
                        </tr>
                        <tr className="border-t border-sage-200">
                          <td className="py-2 font-semibold text-moss-800">Confidence</td>
                          <td className="py-2 text-moss-700">{toPercent(previousSnapshot.confidence, 2)}</td>
                          <td className="py-2 text-moss-700">{toPercent(currentComparison.confidence, 2)}</td>
                        </tr>
                        <tr className="border-t border-sage-200">
                          <td className="py-2 font-semibold text-moss-800">Estimated NTU</td>
                          <td className="py-2 text-moss-700">{String(previousSnapshot.ntu)}</td>
                          <td className="py-2 text-moss-700">{String(currentComparison.ntu)}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            )}
          </div>
        )}
      </section>
    </main>
  )
}

export default App
