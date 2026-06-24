import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './iris.css'
import IrisApp from './IrisApp.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <IrisApp />
  </StrictMode>,
)
