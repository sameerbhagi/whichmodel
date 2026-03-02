import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import ModelCard from '../components/ModelCard'
import { API } from '../config'
import './Dashboard.css'

export default function Dashboard() {
    const [models, setModels] = useState([])
    const [total, setTotal] = useState(0)
    const [providers, setProviders] = useState([])
    const [loading, setLoading] = useState(true)
    const [search, setSearch] = useState('')
    const [providerFilter, setProviderFilter] = useState('')
    const [sortBy, setSortBy] = useState('')
    const [page, setPage] = useState(1)
    const [totalPages, setTotalPages] = useState(1)
    const navigate = useNavigate()

    const perPage = 24

    const fetchModels = useCallback(() => {
        setLoading(true)
        const params = new URLSearchParams({ page: page.toString(), per_page: perPage.toString() })
        if (search) params.set('search', search)
        if (providerFilter) params.set('provider', providerFilter)
        if (sortBy) params.set('sort_by', sortBy)
        fetch(`${API}/models?${params}`)
            .then(r => r.json())
            .then(data => {
                setModels(data.models || [])
                setTotal(data.total || 0)
                setTotalPages(data.total_pages || 1)
                if (data.providers) setProviders(data.providers)
                setLoading(false)
            })
            .catch(() => setLoading(false))
    }, [page, search, providerFilter, sortBy])

    useEffect(() => { fetchModels() }, [fetchModels])

    // Reset page when filters change
    useEffect(() => { setPage(1) }, [search, providerFilter, sortBy])

    // Compute stats from current data set
    const cheapest = models.length ? models.reduce((a, b) => a.input_price_per_1m < b.input_price_per_1m ? a : b) : null
    const largestCtx = models.length ? models.reduce((a, b) => a.context_window > b.context_window ? a : b) : null

    // Debounced search
    const [searchInput, setSearchInput] = useState('')
    useEffect(() => {
        const timer = setTimeout(() => setSearch(searchInput), 300)
        return () => clearTimeout(timer)
    }, [searchInput])

    // Top provider chips (show top 12 by most models)
    const topProviders = ['OpenAI', 'Anthropic', 'Google', 'Meta', 'DeepSeek', 'Mistral AI', 'Qwen (Alibaba)', 'xAI', 'Cohere', 'Amazon', 'NVIDIA', 'Perplexity']
    const displayProviders = providers.filter(p => topProviders.includes(p))

    return (
        <div className="dashboard">
            {/* Hero */}
            <section className="hero">
                <div className="container">
                    <div className="hero__content animate-in">
                        <span className="badge badge-accent hero__badge">Live Data · Powered by OpenRouter</span>
                        <h1 className="hero__title">
                            Find the <span className="gradient-text">Perfect LLM</span> for Your Use Case
                        </h1>
                        <p className="hero__subtitle">
                            Compare {total}+ AI models from {providers.length} providers — with live pricing, context windows,
                            and capabilities. Get personalized recommendations based on your specific needs.
                        </p>
                        <div className="hero__actions">
                            <button className="btn btn-primary btn-lg" onClick={() => navigate('/recommend')}>
                                ✦ Get Recommendation
                            </button>
                            <button className="btn btn-secondary btn-lg" onClick={() => navigate('/compare')}>
                                ⟺ Compare Models
                            </button>
                        </div>
                    </div>

                    {/* Quick Stats */}
                    <div className="hero__stats animate-in animate-delay-2">
                        <div className="quick-stat glass-card">
                            <span className="quick-stat__number">{total}</span>
                            <span className="quick-stat__label">Live Models</span>
                        </div>
                        <div className="quick-stat glass-card">
                            <span className="quick-stat__number">{providers.length}</span>
                            <span className="quick-stat__label">Providers</span>
                        </div>
                        <div className="quick-stat glass-card">
                            <span className="quick-stat__number">{cheapest ? `$${cheapest.input_price_per_1m.toFixed(2)}` : '—'}</span>
                            <span className="quick-stat__label">Cheapest /1M</span>
                        </div>
                        <div className="quick-stat glass-card">
                            <span className="quick-stat__number">{largestCtx ? `${(largestCtx.context_window / 1_000_000).toFixed(0)}M` : '—'}</span>
                            <span className="quick-stat__label">Largest Context</span>
                        </div>
                    </div>
                </div>
                <div className="hero__glow" />
            </section>

            {/* Models Grid */}
            <section className="section">
                <div className="container">
                    <div className="section-header">
                        <div>
                            <h2 className="section-title">Explore Models</h2>
                            <p className="section-desc">
                                {total} models available · Page {page} of {totalPages}
                            </p>
                        </div>
                    </div>

                    {/* Search & Filters */}
                    <div className="dashboard-controls animate-in">
                        <div className="search-bar">
                            <span className="search-icon">🔍</span>
                            <input
                                type="text"
                                className="search-input"
                                placeholder="Search models by name, provider, or description..."
                                value={searchInput}
                                onChange={e => setSearchInput(e.target.value)}
                            />
                            {searchInput && (
                                <button className="search-clear" onClick={() => setSearchInput('')}>✕</button>
                            )}
                        </div>
                        <div className="controls-row">
                            <div className="filter-pills">
                                <button
                                    className={`pill ${!providerFilter ? 'pill--active' : ''}`}
                                    onClick={() => setProviderFilter('')}
                                >
                                    All
                                </button>
                                {displayProviders.map(p => (
                                    <button
                                        key={p}
                                        className={`pill ${providerFilter === p ? 'pill--active' : ''}`}
                                        onClick={() => setProviderFilter(providerFilter === p ? '' : p)}
                                    >
                                        {p}
                                    </button>
                                ))}
                            </div>
                            <select
                                className="sort-select"
                                value={sortBy}
                                onChange={e => setSortBy(e.target.value)}
                            >
                                <option value="">Default Sort</option>
                                <option value="price_asc">Price: Low → High</option>
                                <option value="price_desc">Price: High → Low</option>
                                <option value="context">Context Window</option>
                                <option value="name">Name A–Z</option>
                            </select>
                        </div>
                    </div>

                    {loading ? (
                        <div className="loading">
                            <div className="loading-spinner" />
                            <p>Loading models...</p>
                        </div>
                    ) : models.length === 0 ? (
                        <div className="empty-state">
                            <p>No models found matching your search.</p>
                            <button className="btn btn-secondary" onClick={() => { setSearchInput(''); setProviderFilter(''); }}>
                                Clear Filters
                            </button>
                        </div>
                    ) : (
                        <>
                            <div className="models-grid">
                                {models.map((model, i) => (
                                    <div key={model.id} className="animate-in" style={{ animationDelay: `${Math.min(i * 0.04, 0.5)}s` }}>
                                        <ModelCard model={model} />
                                    </div>
                                ))}
                            </div>

                            {/* Pagination */}
                            {totalPages > 1 && (
                                <div className="pagination">
                                    <button
                                        className="btn btn-secondary btn-sm"
                                        disabled={page <= 1}
                                        onClick={() => setPage(p => Math.max(1, p - 1))}
                                    >
                                        ← Previous
                                    </button>
                                    <div className="pagination-pages">
                                        {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                                            let pageNum
                                            if (totalPages <= 7) {
                                                pageNum = i + 1
                                            } else if (page <= 4) {
                                                pageNum = i + 1
                                            } else if (page >= totalPages - 3) {
                                                pageNum = totalPages - 6 + i
                                            } else {
                                                pageNum = page - 3 + i
                                            }
                                            return (
                                                <button
                                                    key={pageNum}
                                                    className={`page-btn ${pageNum === page ? 'page-btn--active' : ''}`}
                                                    onClick={() => setPage(pageNum)}
                                                >
                                                    {pageNum}
                                                </button>
                                            )
                                        })}
                                    </div>
                                    <button
                                        className="btn btn-secondary btn-sm"
                                        disabled={page >= totalPages}
                                        onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                                    >
                                        Next →
                                    </button>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </section>
        </div>
    )
}
