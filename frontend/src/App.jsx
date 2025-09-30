import { useEffect, useMemo, useState } from "react"

import { DateSelector } from "./components/date-selector"
import { Graph } from "./components/graph"
import { TrendCard } from "./components/trend-card"

const DATA_ROOT = "/data"

async function fetchJson(path) {
  const response = await fetch(path)
  if (!response.ok) {
    const error = new Error(`Échec du chargement de ${path}`)
    error.status = response.status
    throw error
  }
  return response.json()
}

function normalizeDates(list) {
  if (!Array.isArray(list)) return []
  return list
    .map((entry) => (typeof entry === "string" ? entry : entry?.name))
    .filter(Boolean)
}

export default function App() {
  const [dates, setDates] = useState([])
  const [selectedDate, setSelectedDate] = useState()
  const [nodes, setNodes] = useState([])
  const [links, setLinks] = useState([])
  const [trends, setTrends] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false

    async function loadDates() {
      try {
        const list = await fetchJson(`${DATA_ROOT}/list.json`)
        if (cancelled) return
        const normalized = normalizeDates(list)
        setDates(normalized)
        if (normalized.length > 0) {
          setSelectedDate((current) => current ?? normalized[0])
        }
      } catch (err) {
        if (!cancelled) {
          setError(err)
        }
      }
    }

    loadDates()

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!selectedDate) return

    let cancelled = false

    async function loadSnapshot() {
      setLoading(true)
      setError(null)
      try {
        const [nodeData, linkData, trendData] = await Promise.all([
          fetchJson(`${DATA_ROOT}/${selectedDate}/nodes.json`),
          fetchJson(`${DATA_ROOT}/${selectedDate}/edges.json`),
          fetchJson(`${DATA_ROOT}/${selectedDate}/trends.json`).catch(() => []),
        ])
        if (cancelled) return
        setNodes(Array.isArray(nodeData) ? nodeData : [])
        setLinks(Array.isArray(linkData) ? linkData : [])
        setTrends(Array.isArray(trendData) ? trendData : [])
      } catch (err) {
        if (!cancelled) {
          setError(err)
          setNodes([])
          setLinks([])
          setTrends([])
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    loadSnapshot()

    return () => {
      cancelled = true
    }
  }, [selectedDate])

  const graphTitle = useMemo(() => {
    return selectedDate ? `Actualités du ${selectedDate}` : "Actualités"
  }, [selectedDate])

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-[#0c0d11] to-[#06070b] text-foreground">
      <main className="mx-auto flex w-full max-w-7xl flex-col gap-12 px-6 py-10 md:px-10 lg:px-16">
        <header className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div className="space-y-3">
            <p className="text-xs uppercase tracking-[0.6em] text-muted-foreground">Flux RSS Explorer</p>
            <h1 className="text-4xl font-semibold tracking-tight text-white sm:text-5xl">RSS Trends</h1>
            <p className="max-w-xl text-sm text-muted-foreground">
              Naviguez dans le réseau des sujets d’actualité et retrouvez les articles liés pour chaque tendance.
            </p>
          </div>
          <div className="w-full max-w-sm md:max-w-md">
            <DateSelector dates={dates} value={selectedDate} onChange={setSelectedDate} />
          </div>
        </header>

        <section aria-live="polite" className="space-y-4">
          {error ? (
            <div className="rounded-3xl border border-destructive/40 bg-destructive/10 p-6 text-sm text-destructive">
              Impossible de charger les données pour cette date. Vérifiez que les fichiers existent dans le dossier public/data.
            </div>
          ) : (
            <Graph nodes={nodes} links={links} title={graphTitle} />
          )}
          {loading && (
            <p className="text-sm text-muted-foreground">Chargement des données…</p>
          )}
        </section>

        <section className="space-y-6 pb-16">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-semibold text-white">Tendances</h2>
            <span className="text-xs uppercase tracking-[0.35em] text-muted-foreground">
              {trends.length} cartes
            </span>
          </div>
          {trends.length === 0 && !loading ? (
            <p className="text-sm text-muted-foreground">Aucune tendance disponible pour cette date.</p>
          ) : (
            <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
              {trends.map((trend, index) => (
                <TrendCard key={`${trend?.title ?? "trend"}-${index}`} trend={trend} />
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
