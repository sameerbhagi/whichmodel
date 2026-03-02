import React, { useEffect, useState } from 'react';
import './ModelDetailModal.css';

export default function ModelDetailModal() {
    const [model, setModel] = useState(null);
    const [activeTab, setActiveTab] = useState('benchmarks');

    useEffect(() => {
        const handleOpen = (e) => {
            setModel(e.detail);
            setActiveTab('benchmarks');
        };
        window.addEventListener('open-model-detail', handleOpen);
        return () => window.removeEventListener('open-model-detail', handleOpen);
    }, []);

    if (!model) return null;

    const closeModal = () => setModel(null);

    const formatPriceToken = (price) => {
        if (!price || price === 0) return '$0.00';
        return `$${price.toFixed(10)}`;
    };

    return (
        <div className="modal-overlay" onClick={closeModal}>
            <div className="modal-content glass-card" onClick={(e) => e.stopPropagation()}>
                <button className="modal-close" onClick={closeModal}>&times;</button>

                <div className="modal-header">
                    <div className="provider-badge" style={{ background: model.logo_color }}>
                        {model.provider}
                    </div>
                    <h2 className="modal-title">{model.name}</h2>
                    <p className="modal-id">{model.openrouter_id}</p>
                </div>

                <div className="modal-tabs">
                    <button
                        className={`tab-btn ${activeTab === 'benchmarks' ? 'active' : ''}`}
                        onClick={() => setActiveTab('benchmarks')}
                    >
                        Benchmarks
                    </button>
                    <button
                        className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
                        onClick={() => setActiveTab('history')}
                    >
                        History & Data
                    </button>
                    <button
                        className={`tab-btn ${activeTab === 'usecases' ? 'active' : ''}`}
                        onClick={() => setActiveTab('usecases')}
                    >
                        Use Cases
                    </button>
                </div>

                <div className="modal-body">
                    {activeTab === 'benchmarks' && (
                        <div className="tab-pane">
                            <h3>Technical Benchmarks</h3>
                            <div className="benchmark-list">
                                {Object.entries(model.benchmarks || {}).map(([name, score]) => (
                                    <div key={name} className="benchmark-item">
                                        <div className="benchmark-info">
                                            <span className="benchmark-name">{name}</span>
                                            <span className="benchmark-score">{score}%</span>
                                        </div>
                                        <div className="benchmark-bar-bg">
                                            <div className="benchmark-bar-fill" style={{ width: `${score}%` }}></div>
                                        </div>
                                    </div>
                                ))}
                                {(!model.benchmarks || Object.keys(model.benchmarks).length === 0) && (
                                    <p className="empty-state">No detailed benchmark data available for this model yet.</p>
                                )}
                            </div>
                        </div>
                    )}

                    {activeTab === 'history' && (
                        <div className="tab-pane">
                            <div className="data-section">
                                <h3>Launch Information</h3>
                                <div className="data-grid">
                                    <div className="data-card">
                                        <span className="label">Release Date</span>
                                        <span className="value">{model.release_date || 'Unknown'}</span>
                                    </div>
                                    <div className="data-card">
                                        <span className="label">Context Window</span>
                                        <span className="value">{model.context_window.toLocaleString()} tokens</span>
                                    </div>
                                    <div className="data-card">
                                        <span className="label">Input Price</span>
                                        <span className="value">{formatPriceToken(model.input_price_per_token)}/token</span>
                                    </div>
                                    <div className="data-card">
                                        <span className="label">Output Price</span>
                                        <span className="value">{formatPriceToken(model.output_price_per_token)}/token</span>
                                    </div>
                                </div>
                            </div>

                            <div className="data-section">
                                <h3>Update History</h3>
                                <ul className="history-list">
                                    {(model.update_history || []).map((item, i) => (
                                        <li key={i}>{item}</li>
                                    ))}
                                    {(!model.update_history || model.update_history.length === 0) && (
                                        <li>Initial model release via OpenRouter.</li>
                                    )}
                                </ul>
                            </div>
                        </div>
                    )}

                    {activeTab === 'usecases' && (
                        <div className="tab-pane">
                            <h3>Recommended Use Cases</h3>
                            <div className="usecase-grid">
                                {(model.use_cases || []).map((uc, i) => (
                                    <div key={i} className="usecase-tag">{uc}</div>
                                ))}
                                {(!model.use_cases || model.use_cases.length === 0) && (
                                    <p className="empty-state">General purpose LLM suitable for various tasks.</p>
                                )}
                            </div>

                            <h3>Strengths</h3>
                            <div className="strengths-weaknesses">
                                <div className="list-section">
                                    <ul className="strength-list">
                                        {(model.strengths || []).map((s, i) => <li key={i}>{s}</li>)}
                                    </ul>
                                </div>
                                <div className="list-section">
                                    <h4>Considerations</h4>
                                    <ul className="weakness-list">
                                        {(model.weaknesses || []).map((w, i) => <li key={i}>{w}</li>)}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                <div className="modal-footer">
                    <p className="footer-note">Data source: OpenRouter & WhichModel Curated Datasets (Feb 2026)</p>
                    <button className="btn btn-primary" onClick={closeModal}>Close Details</button>
                </div>
            </div>
        </div>
    );
}
