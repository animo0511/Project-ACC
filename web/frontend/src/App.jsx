import { useState } from 'react'
import './App.css'

function App() {
    const [count, setCount] = useState(0)

    return (
        <>
            <div className="card">
                <h1>Computer Vision App</h1>
                <p>
                    Frontend environment is ready.
                </p>
            </div>
        </>
    )
}

export default App
