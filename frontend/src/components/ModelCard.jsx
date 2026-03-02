import './ModelCard.css'

export default function ModelCard({ model, onCompare, isSelected }) {
    const formatContext = (tokens) => {
        if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(tokens % 1000000 === 0 ? 0 : 1)}M`
        return `${(tokens / 1000).toFixed(0)}K`
    }

    const formatPriceToken = (price) => {
        if (!price || price === 0) return '$0.00'
        // High precision for per-token prices
        return `$${price.toFixed(8)}`
    }

    // Get the top benchmark score
    const benchmarks = Object.entries(model.benchmarks || {})
    const avgBenchmark = benchmarks.length
        ? (benchmarks.reduce((s, [, v]) => s + v, 0) / benchmarks.length).toFixed(1)
        : null

    const isMultimodal = model.input_modalities?.includes('image') || model.input_modalities?.includes('video')
    const canGenImages = model.output_modalities?.includes('image')

    return (
        <div
            className={`model-card glass-card ${isSelected ? 'model-card--selected' : ''}`}
            onClick={() => window.dispatchEvent(new CustomEvent('open-model-detail', { detail: model }))}
            style={{ cursor: 'pointer' }}
        >
            <div className="model-card__header">
                <div className="model-card__provider-dot" style={{ background: model.logo_color }} />
                <div>
                    <h3 className="model-card__name">{model.name}</h3>
                    <span className="model-card__provider">{model.provider}</span>
                </div>
            </div>

            <p className="model-card__desc">{model.description || 'No description available.'}</p>

            <div className="model-card__stats">
                <div className="stat">
                    <span className="stat__label">Context</span>
                    <span className="stat__value">{formatContext(model.context_window)}</span>
                </div>
                <div className="stat">
                    <span className="stat__label">Input</span>
                    <span className="stat__value" title={`${formatPriceToken(model.input_price_per_token)} per token`}>
                        {formatPriceToken(model.input_price_per_token)}
                    </span>
                    <span className="stat__sublabel">/ token</span>
                </div>
                <div className="stat">
                    <span className="stat__label">Output</span>
                    <span className="stat__value" title={`${formatPriceToken(model.output_price_per_token)} per token`}>
                        {formatPriceToken(model.output_price_per_token)}
                    </span>
                    <span className="stat__sublabel">/ token</span>
                </div>
                {avgBenchmark && (
                    <div className="stat">
                        <span className="stat__label">Avg Score</span>
                        <span className="stat__value">{avgBenchmark}</span>
                    </div>
                )}
            </div>

            {/* Modality badges */}
            <div className="model-card__tags">
                {isMultimodal && <span className="badge badge-info">📷 Multimodal</span>}
                {canGenImages && <span className="badge badge-accent">🖼️ Image Gen</span>}
                {model.strengths?.slice(0, 2).map((s, i) => (
                    <span key={i} className="badge badge-success">{s}</span>
                ))}
            </div>

            {onCompare && (
                <button
                    className={`btn btn-sm ${isSelected ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => onCompare(model.id)}
                >
                    {isSelected ? '✓ Selected' : '+ Compare'}
                </button>
            )}
        </div>
    )
}
