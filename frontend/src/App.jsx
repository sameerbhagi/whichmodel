import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import ModelDetailModal from './components/ModelDetailModal'
import Dashboard from './pages/Dashboard'
import Compare from './pages/Compare'
import Recommend from './pages/Recommend'
import './App.css'

function App() {
  return (
    <div className="app">
      <Navbar />
      <main className="page-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/recommend" element={<Recommend />} />
        </Routes>
      </main>
      <ModelDetailModal />
    </div>
  )
}

export default App
