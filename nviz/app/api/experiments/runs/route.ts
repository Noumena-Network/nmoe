import { NextRequest, NextResponse } from 'next/server'
import { listRunsForExperiment } from '@/lib/server/experiments'
import { selectedRuns } from '@/lib/server/duckdb'

export async function GET(req: NextRequest) {
  const experiment_id = req.nextUrl.searchParams.get('experiment_id')
  const limit = Math.max(1, Math.min(parseInt(req.nextUrl.searchParams.get('limit') || '200', 10), 2000))

  try {
    const runs = listRunsForExperiment(experiment_id || undefined, limit)
    const metrics = await selectedRuns(runs.map((r) => r.run))
    const byRun = new Map(metrics.map((m) => [m.run, m]))

    const out = runs.map((r) => {
      const m = byRun.get(r.run)
      return {
        ...r,
        last_ts: m?.last_ts ?? 0,
        last_step: m?.last_step ?? 0,
      }
    })
    return NextResponse.json({ runs: out }, { headers: { 'Cache-Control': 'private, max-age=5' } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}
