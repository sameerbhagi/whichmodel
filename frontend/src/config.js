/**
 * API configuration — picks up the backend URL from environment variables.
 * In development:  VITE_API_URL defaults to http://localhost:8001/api
 * In production:   Set VITE_API_URL in your hosting provider (e.g. Vercel env vars)
 */
export const API = import.meta.env.VITE_API_URL || 'http://localhost:8001/api';
