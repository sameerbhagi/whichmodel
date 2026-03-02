import { useState, useEffect } from 'react'
import { API } from '../config'
import './Compare.css'

const BENCHMARK_LABELS = {
    'MMLU': 'MMLU (Knowledge)',
    'HumanEval': 'HumanEval (Code)',
    'MATH': 'MATH (Math)',
    'GPQA': 'GPQA (Science)',
    'MGSM': 'MGSM (Multilingual)',
    'ARC-Challenge': 'ARC (Reasoning)',
}

export default function Compare() {
    const [allModels, setAllModels] = useState([])
    const [selected, setSelected] = useState([])
    const [comparison, setComparison] = useState(null)
    const [loading, setLoading] = useState(true)
    const [searchTerm, setSearchTerm] = useState('')
    const [dropdownOpen, setDropdownOpen] = useState(false)

    useEffect(() => {
        fetch(`${API}/models?per_page=200`)
            .then(r => r.json())
            .then(data => {
                setAllModels(data.models || [])
                setLoading(false)
            })
            .catch(() => setLoading(false))
    }, [])

    const toggleModel = (id) => {
        setSelected(prev =>
            prev.includes(id) ? prev.filter(x => x !== id) : prev.length < 6 ? [...prev, id] : prev
        )
    }

    const removeModel = (id) => {
        setSelected(prev => prev.filter(x => x !== id))
        setComparison(null)
    }

    const doCompare = () => {
        if (selected.length < 2) return
        fetch(`${API}/models/compare?ids=${selected.join(',')}`)
            .then(r => r.json())
            .then(data => setComparison(data))
    }

    const formatContext = (tokens) => {
        if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(tokens % 1000000 === 0 ? 0 : 1)}M`
        return `${(tokens / 1000).toFixed(0)}K`
    }

    const getBenchmarkColor = (score, allScores) => {
        const valid = allScores.filter(s => s > 0)
        if (valid.length < 2) return ''
        const max = Math.max(...valid)
        const min = Math.min(...valid)
        if (score === max) return 'cell--best'
        if (score === min && valid.length > 2) return 'cell--worst'
        return ''
    }

    const getPriceColor = (price, allPrices) => {
        const min = Math.min(...allPrices)
        const max = Math.max(...allPrices)
        if (price === min) return 'cell--best'
        if (price === max && allPrices.length > 2) return 'cell--worst'
        return ''
    }

    // Filter models for search dropdown
    const filteredModels = allModels.filter(m => {
        const q = searchTerm.toLowerCase()
        return (m.name.toLowerCase().includes(q) || m.provider.toLowerCase().includes(q)) && !selected.includes(m.id)
    }).slice(0, 20)

    // Selected model objects
    const selectedModels = selected.map(id => allModels.find(m => m.id === id)).filter(Boolean)

    // Get benchmarks actually present in compared models
    const activeBenchmarks = comparison
        ? Object.entries(BENCHMARK_LABELS).filter(([key]) =>
            comparison.models.some(m => m.benchmarks?.[key] !== undefined))
        : []

    return (
        <div className="compare-page">
            <div className="container">
                <div className="compare-header animate-in">
                    <h1 className="compare-title">
                        <span className="gradient-text">Compare</span> Models
                    </h1>
                    <p className="compare-desc">
                        Select 2–6 models to compare side-by-side across pricing, specs, and benchmarks.
                    </p>
                </div>

                {/* Searchable Model Selector */}
                <div className="model-selector animate-in animate-delay-1">
                    <div className="model-search-wrap">
                        <div className="model-search-input-wrap">
                            <span className="search-icon">🔍</span>
                            <input
                                type="text"
                                className="model-search-input"
                                placeholder="Search models to add..."
                                value={searchTerm}
                                onChange={e => { setSearchTerm(e.target.value); setDropdownOpen(true) }}
                                onFocus={() => setDropdownOpen(true)}
                            />
                        </div>
                        {dropdownOpen && searchTerm && (
                            <div className="model-dropdown">
                                {filteredModels.length === 0 ? (
                                    <div className="dropdown-empty">No models found</div>
                                ) : (
                                    filteredModels.map(m => (
                                        <button
                                            key={m.id}
                                            className="dropdown-item"
                                            onClick={() => {
                                                toggleModel(m.id)
                                                setSearchTerm('')
                                                setDropdownOpen(false)
                                            }}
                                        >
                                            <span className="chip-dot" style={{ background: m.logo_color }} />
                                            <div className="dropdown-item-info">
                                                <span className="dropdown-item-name">{m.name}</span>
                                                <span className="dropdown-item-provider">{m.provider} · ${m.input_price_per_1m.toFixed(2)}/1M · {formatContext(m.context_window)}</span>
                                            </div>
                                        </button>
                                    ))
                                )}
                            </div>
                        )}
                    </div>

                    {/* Selected chips */}
                    {selectedModels.length > 0 && (
                        <div className="selected-chips">
                            {selectedModels.map(m => (
                                <div key={m.id} className="selector-chip selector-chip--active">
                                    <span className="chip-dot" style={{ background: m.logo_color }} />
                                    <span className="chip-name">{m.name}</span>
                                    <button className="chip-remove" onClick={() => removeModel(m.id)}>✕</button>
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="selector-actions">
                        <span className="selector-count">{selected.length} selected{selected.length < 2 && ' (min 2)'}</span>
                        <button
                            className="btn btn-primary"
                            disabled={selected.length < 2}
                            onClick={doCompare}
                        >
                            Compare {selected.length >= 2 ? `(${selected.length})` : ''}
                        </button>
                    </div>
                </div>

                {/* Quick Pick Suggestions */}
                {selected.length === 0 && !loading && (
                    <div className="quick-picks animate-in animate-delay-2">
                        <p className="quick-picks-label">Popular comparisons:</p>
                        <div className="quick-pick-suggestions">
                            {[
                                { label: 'GPT-4o vs Claude 3.5 Sonnet', ids: ['openai-gpt-4o', 'anthropic-claude-3.5-sonnet'] },
                                { label: 'Gemini 2.5 Pro vs GPT-4o', ids: ['google-gemini-2.5-pro', 'openai-gpt-4o'] },
                                { label: 'DeepSeek R1 vs o1', ids: ['deepseek-deepseek-r1', 'openai-o1'] },
                            ].map(({ label, ids }) => (
                                <button
                                    key={label}
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => { setSelected(ids); }}
                                >
                                    {label}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Comparison Table */}
                {comparison && (
                    <div className="comparison-result animate-in">
                        <div className="compare-table-wrap glass-card">
                            <div className="compare-table-scroll">
                                <table className="compare-table">
                                    <thead>
                                        <tr>
                                            <th className="compare-table__label-col">Metric</th>
                                            {comparison.models.map(m => (
                                                <th key={m.id} className="compare-table__model-col">
                                                    <div className="th-model">
                                                        <span className="th-dot" style={{ background: m.logo_color }} />
                                                        <div>
                                                            <div className="th-name">{m.name}</div>
                                                            <div className="th-provider">{m.provider}</div>
                                                        </div>
                                                    </div>
                                                </th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {/* Pricing rows */}
                                        <tr className="section-row"><td colSpan={comparison.models.length + 1}>Pricing (per 1M tokens)</td></tr>
                                        <tr>
                                            <td className="label-cell">Input Price</td>
                                            {comparison.models.map(m => {
                                                const allPrices = comparison.models.map(x => x.input_price_per_1m)
                                                return (
                                                    <td key={m.id} className={`value-cell ${getPriceColor(m.input_price_per_1m, allPrices)}`}>
                                                        ${m.input_price_per_1m.toFixed(2)}
                                                    </td>
                                                )
                                            })}
                                        </tr>
                                        <tr>
                                            <td className="label-cell">Output Price</td>
                                            {comparison.models.map(m => {
                                                const allPrices = comparison.models.map(x => x.output_price_per_1m)
                                                return (
                                                    <td key={m.id} className={`value-cell ${getPriceColor(m.output_price_per_1m, allPrices)}`}>
                                                        ${m.output_price_per_1m.toFixed(2)}
                                                    </td>
                                                )
                                            })}
                                        </tr>

                                        {/* Specs rows */}
                                        <tr className="section-row"><td colSpan={comparison.models.length + 1}>Specifications</td></tr>
                                        <tr>
                                            <td className="label-cell">Context Window</td>
                                            {comparison.models.map(m => {
                                                const allCtx = comparison.models.map(x => x.context_window)
                                                const isMax = m.context_window === Math.max(...allCtx)
                                                return (
                                                    <td key={m.id} className={`value-cell ${isMax ? 'cell--best' : ''}`}>
                                                        {formatContext(m.context_window)}
                                                    </td>
                                                )
                                            })}
                                        </tr>
                                        <tr>
                                            <td className="label-cell">Max Output</td>
                                            {comparison.models.map(m => (
                                                <td key={m.id} className="value-cell">{formatContext(m.max_output_tokens)}</td>
                                            ))}
                                        </tr>
                                        <tr>
                                            <td className="label-cell">Modality</td>
                                            {comparison.models.map(m => (
                                                <td key={m.id} className="value-cell modality-cell">
                                                    {m.input_modalities?.includes('image') && <span className="badge badge-info">📷 Image</span>}
                                                    {m.input_modalities?.includes('video') && <span className="badge badge-info">🎥 Video</span>}
                                                    {m.output_modalities?.includes('image') && <span className="badge badge-accent">🖼️ Image Gen</span>}
                                                    {!m.input_modalities?.includes('image') && !m.input_modalities?.includes('video') && (
                                                        <span className="badge badge-muted">Text Only</span>
                                                    )}
                                                </td>
                                            ))}
                                        </tr>

                                        {/* Benchmark rows — only show if any models have data */}
                                        {activeBenchmarks.length > 0 && (
                                            <>
                                                <tr className="section-row"><td colSpan={comparison.models.length + 1}>Benchmarks</td></tr>
                                                {activeBenchmarks.map(([key, label]) => (
                                                    <tr key={key}>
                                                        <td className="label-cell">{label}</td>
                                                        {comparison.models.map(m => {
                                                            const score = m.benchmarks?.[key]
                                                            const allScores = comparison.models.map(x => x.benchmarks?.[key] || 0)
                                                            return (
                                                                <td key={m.id} className={`value-cell ${score ? getBenchmarkColor(score, allScores) : ''}`}>
                                                                    {score ? (
                                                                        <div className="score-cell">
                                                                            <span>{score.toFixed(1)}</span>
                                                                            <div className="score-bar">
                                                                                <div className="score-bar__fill" style={{ width: `${score}%` }} />
                                                                            </div>
                                                                        </div>
                                                                    ) : <span className="no-data">—</span>}
                                                                </td>
                                                            )
                                                        })}
                                                    </tr>
                                                ))}
                                            </>
                                        )}

                                        {/* Strengths */}
                                        {comparison.models.some(m => m.strengths?.length > 0) && (
                                            <>
                                                <tr className="section-row"><td colSpan={comparison.models.length + 1}>Strengths & Weaknesses</td></tr>
                                                <tr>
                                                    <td className="label-cell">Strengths</td>
                                                    {comparison.models.map(m => (
                                                        <td key={m.id} className="value-cell strengths-cell">
                                                            {m.strengths?.length > 0 ? m.strengths.map((s, i) => (
                                                                <span key={i} className="badge badge-success">{s}</span>
                                                            )) : <span className="no-data">—</span>}
                                                        </td>
                                                    ))}
                                                </tr>
                                                <tr>
                                                    <td className="label-cell">Weaknesses</td>
                                                    {comparison.models.map(m => (
                                                        <td key={m.id} className="value-cell strengths-cell">
                                                            {m.weaknesses?.length > 0 ? m.weaknesses.map((w, i) => (
                                                                <span key={i} className="badge badge-warning">{w}</span>
                                                            )) : <span className="no-data">—</span>}
                                                        </td>
                                                    ))}
                                                </tr>
                                            </>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        {/* Benchmark Chart — only show if benchmarks exist */}
                        {activeBenchmarks.length > 0 && (
                            <div className="benchmark-chart glass-card">
                                <h3 className="chart-title">Benchmark Comparison</h3>
                                <div className="chart-grid">
                                    {activeBenchmarks.map(([key, label]) => (
                                        <div key={key} className="chart-benchmark">
                                            <div className="chart-benchmark__label">{label.split(' (')[0]}</div>
                                            <div className="chart-benchmark__bars">
                                                {comparison.models.map(m => {
                                                    const score = m.benchmarks?.[key] || 0
                                                    return (
                                                        <div key={m.id} className="bar-row">
                                                            <span className="bar-model">{m.name.length > 15 ? m.name.slice(0, 15) + '…' : m.name}</span>
                                                            <div className="bar-track">
                                                                <div
                                                                    className="bar-fill"
                                                                    style={{
                                                                        width: `${score}%`,
                                                                        background: m.logo_color,
                                                                    }}
                                                                />
                                                            </div>
                                                            <span className="bar-score">{score ? score.toFixed(1) : '—'}</span>
                                                        </div>
                                                    )
                                                })}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    )
}
