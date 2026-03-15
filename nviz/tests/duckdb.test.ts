import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { mkdtempSync, rmSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import os from 'node:os'

import { DuckDBInstance } from '@duckdb/node-api'
import { allRuns, scalarsSampled, selectedRuns } from '@/lib/server/duckdb'

let dir: string
let runDir: string
let runDirB: string

beforeAll(() => {
  dir = mkdtempSync(join(os.tmpdir(), 'nviz-metrics-'))
  process.env.NVIZ_METRICS_DIR = dir
  runDir = join(dir, 'runA')
  runDirB = join(dir, 'runB')
  mkdirSync(runDir, { recursive: true })
  mkdirSync(runDirB, { recursive: true })
})

afterAll(() => {
  try { rmSync(dir, { recursive: true, force: true }) } catch {}
})

describe('duckdb layer', () => {
  beforeAll(async () => {
    const sqlLit = (s: string) => `'${s.replaceAll("'", "''")}'`
    const inst = await DuckDBInstance.create(':memory:')
    const conn = await inst.connect()
    try {
      const now = Date.now()
      await conn.run(
        `COPY (SELECT 'runA' AS run, 'train/loss' AS tag, 1::INTEGER AS step, ${now - 3000}::BIGINT AS ts_ms, 10.0::DOUBLE AS value) TO ${sqlLit(join(runDir, 'step_00000001.parquet'))} (FORMAT PARQUET)`
      )
      await conn.run(
        `COPY (SELECT 'runA' AS run, 'train/loss' AS tag, 2::INTEGER AS step, ${now - 2000}::BIGINT AS ts_ms, 7.5::DOUBLE AS value) TO ${sqlLit(join(runDir, 'step_00000002.parquet'))} (FORMAT PARQUET)`
      )
      await conn.run(
        `COPY (SELECT 'runA' AS run, 'train/loss' AS tag, 3::INTEGER AS step, ${now - 1000}::BIGINT AS ts_ms, 6.2::DOUBLE AS value) TO ${sqlLit(join(runDir, 'step_00000003.parquet'))} (FORMAT PARQUET)`
      )
    } finally {
      try { await conn.close() } catch {}
      try { await inst.close() } catch {}
    }
  })

  it('lists runs sorted by recency', async () => {
    const runs = await allRuns()
    expect(runs.length).toBe(1)
    expect(runs[0].run).toBe('runA')
    expect(runs[0].last_step).toBe(3)
    expect(runs[0].last_ts).toBeGreaterThan(0)
  })

  it('summarizes only requested runs without scanning unrelated runs', async () => {
    const sqlLit = (s: string) => `'${s.replaceAll("'", "''")}'`
    const inst = await DuckDBInstance.create(':memory:')
    const conn = await inst.connect()
    try {
      await conn.run(
        `COPY (SELECT 'runB' AS run, 'train/loss' AS tag, 9::INTEGER AS step, ${Date.now()}::BIGINT AS ts_ms, 4.2::DOUBLE AS value) TO ${sqlLit(join(runDirB, 'step_00000009.parquet'))} (FORMAT PARQUET)`
      )
    } finally {
      try { await conn.close() } catch {}
      try { await inst.close() } catch {}
    }

    const runs = await selectedRuns(['runA'])
    expect(runs.length).toBe(1)
    expect(runs[0].run).toBe('runA')
    expect(runs[0].last_step).toBe(3)
  })

  it('returns scalar series (sampled) preserving endpoints', async () => {
    const s = await scalarsSampled('runA','train/loss', 16)
    expect(s.length).toBeGreaterThanOrEqual(2)
    expect(s[0].step).toBe(1)
    expect(s[s.length-1].step).toBe(3)
  })
  
  it('requires NVIZ_METRICS_DIR to be set', async () => {
    const prev = process.env.NVIZ_METRICS_DIR
    // unset and expect throws when scanning
    // @ts-ignore
    delete process.env.NVIZ_METRICS_DIR
    await expect(async () => await allRuns()).toThrow()
    process.env.NVIZ_METRICS_DIR = prev
  })
})
