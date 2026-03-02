import { NavLink } from 'react-router-dom'
import './Navbar.css'

export default function Navbar() {
    return (
        <nav className="navbar">
            <div className="container navbar-inner">
                <NavLink to="/" className="navbar-brand">
                    <div className="navbar-logo">
                        <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                            <rect width="28" height="28" rx="8" fill="url(#logo-grad)" />
                            <path d="M8 14L12 10L16 14L20 10" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                            <path d="M8 18L12 14L16 18L20 14" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.5" />
                            <defs>
                                <linearGradient id="logo-grad" x1="0" y1="0" x2="28" y2="28">
                                    <stop stopColor="#6366f1" />
                                    <stop offset="1" stopColor="#a78bfa" />
                                </linearGradient>
                            </defs>
                        </svg>
                        <span className="navbar-title">WhichModel</span>
                    </div>
                </NavLink>

                <div className="navbar-links">
                    <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} end>
                        <span className="nav-icon">◆</span> Dashboard
                    </NavLink>
                    <NavLink to="/compare" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                        <span className="nav-icon">⟺</span> Compare
                    </NavLink>
                    <NavLink to="/recommend" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                        <span className="nav-icon">✦</span> Recommend
                    </NavLink>
                </div>
            </div>
        </nav>
    )
}
