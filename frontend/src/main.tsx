import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { ActivationGate } from './components/ActivationGate'
import './styles.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode><ActivationGate><App /></ActivationGate></StrictMode>,
)

