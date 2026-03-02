import { useState } from 'react'
import { API } from '../config'
import './Recommend.css'

const PRIORITY_DIMS = [
    { key: 'quality', label: 'Overall Quality', icon: '◆' },
    { key: 'cost', label: 'Cost Efficiency', icon: '💰' },
    { key: 'speed', label: 'Speed / Latency', icon: '⚡' },
    { key: 'code', label: 'Code Generation', icon: '🖥' },
    { key: 'reasoning', label: 'Reasoning / Logic', icon: '🧠' },
    { key: 'long_context', label: 'Long Context', icon: '📄' },
]

export default function Recommend() {
    const [useCase, setUseCase] = useState('')
    const [priorities, setPriorities] = useState({
        quality: 3, cost: 3, speed: 3, code: 3, reasoning: 3, long_context: 3,
    })
    const [budget, setBudget] = useState('')
    const [minContext, setMinContext] = useState('')
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)

    const updatePriority = (key, val) => {
        setPriorities(prev => ({ ...prev, [key]: parseInt(val) }))
    }

    const submit = async () => {
        if (useCase.length < 10) return
        setLoading(true)
        setResult(null)
        try {
            const body = {
                use_case: useCase,
                priorities,
                budget_per_1m_tokens: budget ? parseFloat(budget) : null,
                min_context_window: minContext ? parseInt(minContext) : null,
            }
            const res = await fetch(`${API}/recommend`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            })
            const data = await res.json()
            setResult(data)
        } catch (e) {
            console.error(e)
        }
        setLoading(false)
    }

    const ScoreRing = ({ score, size = 80 }) => {
        const radius = (size - 8) / 2
        const circumference = 2 * Math.PI * radius
        const offset = circumference - (score / 100) * circumference
        const color = score >= 80 ? 'var(--accent-success)' : score >= 60 ? 'var(--accent-warning)' : 'var(--accent-danger)'

        return (
            <div className="score-ring" style={{ width: size, height: size }}>
                <svg width={size} height={size}>
                    <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="var(--bg-tertiary)" strokeWidth="4" />
                    <circle
                        cx={size / 2} cy={size / 2} r={radius} fill="none"
                        stroke={color} strokeWidth="4" strokeLinecap="round"
                        strokeDasharray={circumference}
                        strokeDashoffset={offset}
                        transform={`rotate(-90 ${size / 2} ${size / 2})`}
                        style={{ transition: 'stroke-dashoffset 1s ease' }}
                    />
                </svg>
                <div className="score-ring__value" style={{ color }}>{score.toFixed(0)}</div>
            </div>
        )
    }

    return (
        <div className="recommend-page">
            <div className="container">
                <div className="recommend-header animate-in">
                    <span className="badge badge-accent">Personalized</span>
                    <h1 className="recommend-title">
                        <span className="gradient-text">Which Model</span> Should You Use?
                    </h1>
                    <p className="recommend-desc">
                        Describe your use case and set your priorities. Our engine will analyze all models
                        and recommend the best match with detailed cost breakdowns and reasoning.
                    </p>
                </div>

                <div className="recommend-layout">
                    {/* Input Form */}
                    <div className="recommend-form glass-card animate-in animate-delay-1">
                        <div className="form-group">
                            <label className="label">Describe Your Use Case</label>
                            <textarea
                                className="input"
                                placeholder="e.g. I need to build a coding assistant that helps developers write Python code, debug issues, and generate unit tests. It should be fast and cost-effective for high-volume API calls..."
                                value={useCase}
                                onChange={e => setUseCase(e.target.value)}
                                rows={5}
                            />
                            <span className="char-count">{useCase.length} characters</span>
                        </div>

                        <div className="form-group">
                            <label className="label">Priority Weights</label>
                            <div className="priority-grid">
                                {PRIORITY_DIMS.map(dim => (
                                    <div key={dim.key} className="slider-container">
                                        <div className="slider-header">
                                            <span className="slider-label">{dim.icon} {dim.label}</span>
                                            <span className="slider-value">{priorities[dim.key]}/5</span>
                                        </div>
                                        <input
                                            type="range"
                                            min={1}
                                            max={5}
                                            value={priorities[dim.key]}
                                            onChange={e => updatePriority(dim.key, e.target.value)}
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="form-row">
                            <div className="form-group">
                                <label className="label">Max Budget ($/1M tokens)</label>
                                <input
                                    className="input"
                                    type="number"
                                    placeholder="No limit"
                                    value={budget}
                                    onChange={e => setBudget(e.target.value)}
                                />
                            </div>
                            <div className="form-group">
                                <label className="label">Min Context Window</label>
                                <input
                                    className="input"
                                    type="number"
                                    placeholder="No minimum"
                                    value={minContext}
                                    onChange={e => setMinContext(e.target.value)}
                                />
                            </div>
                        </div>

                        <button
                            className="btn btn-primary btn-lg submit-btn"
                            onClick={submit}
                            disabled={useCase.length < 10 || loading}
                        >
                            {loading ? (
                                <>
                                    <span className="loading-spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                                    Analyzing...
                                </>
                            ) : (
                                <>✦ Find My Model</>
                            )}
                        </button>
                    </div>

                    {/* Results */}
                    {result && (
                        <div className="recommend-results animate-in">
                            {/* Top Pick */}
                            <div className="top-pick glass-card">
                                <div className="top-pick__header">
                                    <span className="badge badge-success">🏆 Top Recommendation</span>
                                    <ScoreRing score={result.top_pick.match_percentage} size={90} />
                                </div>
                                <div className="top-pick__body">
                                    <div className="top-pick__model-info">
                                        <div className="top-pick__dot" style={{ background: result.top_pick.model.logo_color }} />
                                        <div>
                                            <h2 className="top-pick__name">{result.top_pick.model.name}</h2>
                                            <span className="top-pick__provider">{result.top_pick.model.provider}</span>
                                        </div>
                                    </div>
                                    <p className="top-pick__desc">{result.top_pick.model.description}</p>

                                    <div className="top-pick__dimensions">
                                        {Object.entries(result.top_pick.dimension_scores).map(([dim, score]) => (
                                            <div key={dim} className="dim-bar">
                                                <div className="dim-bar__header">
                                                    <span className="dim-bar__label">{dim.replace('_', ' ')}</span>
                                                    <span className="dim-bar__value">{score.toFixed(0)}</span>
                                                </div>
                                                <div className="dim-bar__track">
                                                    <div
                                                        className="dim-bar__fill"
                                                        style={{
                                                            width: `${score}%`,
                                                            background: score >= 80 ? 'var(--accent-success)' : score >= 60 ? 'var(--accent-primary)' : 'var(--accent-warning)'
                                                        }}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                    </div>

                                    <div className="top-pick__details">
                                        <div className="detail-card">
                                            <span className="detail-card__icon">💡</span>
                                            <div>
                                                <h4 className="detail-card__title">Why This Model?</h4>
                                                <p className="detail-card__text">{result.top_pick.reasoning}</p>
                                            </div>
                                        </div>
                                        <div className="detail-card">
                                            <span className="detail-card__icon">💰</span>
                                            <div>
                                                <h4 className="detail-card__title">Cost Estimate</h4>
                                                <p className="detail-card__text">{result.top_pick.cost_estimate}</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="top-pick__tags">
                                        <div>
                                            <span className="tag-label">Strengths</span>
                                            <div className="tag-list">
                                                {result.top_pick.model.strengths?.map((s, i) => (
                                                    <span key={i} className="badge badge-success">{s}</span>
                                                ))}
                                            </div>
                                        </div>
                                        <div>
                                            <span className="tag-label">Considerations</span>
                                            <div className="tag-list">
                                                {result.top_pick.model.weaknesses?.map((w, i) => (
                                                    <span key={i} className="badge badge-warning">{w}</span>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Alternatives */}
                            {result.alternatives?.length > 0 && (
                                <div className="alternatives">
                                    <h3 className="alternatives__title">Other Strong Options</h3>
                                    <div className="alternatives__grid">
                                        {result.alternatives.map((alt, i) => (
                                            <div key={i} className="alt-card glass-card">
                                                <div className="alt-card__header">
                                                    <div className="alt-card__model">
                                                        <div className="alt-card__dot" style={{ background: alt.model.logo_color }} />
                                                        <div>
                                                            <h4 className="alt-card__name">{alt.model.name}</h4>
                                                            <span className="alt-card__provider">{alt.model.provider}</span>
                                                        </div>
                                                    </div>
                                                    <ScoreRing score={alt.match_percentage} size={56} />
                                                </div>
                                                <p className="alt-card__reasoning">{alt.reasoning}</p>
                                                <div className="alt-card__cost">
                                                    <span className="alt-card__cost-label">Cost</span>
                                                    <span className="alt-card__cost-value">
                                                        ${alt.model.input_price_per_1m}/M in · ${alt.model.output_price_per_1m}/M out
                                                    </span>
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
        </div>
    )
}
