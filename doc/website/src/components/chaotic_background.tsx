"use client"

import { useEffect, useRef } from "react"

export function ChaoticBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas to full width/height
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // Lorenz attractor parameters
    const sigma = 22
    const rho = 58
    const beta = 15 / 3

    // Initial conditions
    const points: Array<{ x: number; y: number; z: number }> = []

    // Create multiple starting points for more visual interest
    const nPoints=5
    for (let i = 0; i < nPoints; i++) {
      points.push({
        x: Math.random() * 0.1,
        y: Math.random() * 0.1,
        z: Math.random() * 0.1,
      })
    }

    // Trail length
    const maxTrail = 100
    const trails: Array<Array<{ x: number; y: number }>> = points.map(() => [])

    // Colors
    const colors = [
      "rgba(16, 185, 129, 0.5)", // emerald
      "rgba(6, 182, 212, 0.5)", // cyan
      "rgba(59, 130, 246, 0.5)", // blue
      "rgba(139, 92, 246, 0.5)", // violet
      "rgba(236, 72, 153, 0.5)", // pink
    ]

    // Animation parameters
    const dt = 0.005
    const scale = 10

    // Animation function
    const animate = () => {
      // Fade out previous frame
      ctx.fillStyle = "rgba(255, 255, 255, 0.05)"
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Update each point
      points.forEach((point, idx) => {
        // Lorenz equations
        const dx = sigma * (point.y - point.x) * dt
        const dy = (point.x * (rho - point.z) - point.y) * dt
        const dz = (point.x * point.y - beta * point.z) * dt

        point.x += dx
        point.y += dy
        point.z += dz

        // Project 3D to 2D
        const screenX = canvas.width / 2 + point.x * scale
        const screenY = canvas.height / 2 + point.y * scale

        // Add to trail
        trails[idx].push({ x: screenX, y: screenY })
        if (trails[idx].length > maxTrail) {
          trails[idx].shift()
        }

        // Draw trail
        if (trails[idx].length > 1) {
          ctx.beginPath()
          ctx.moveTo(trails[idx][0].x, trails[idx][0].y)

          for (let i = 1; i < trails[idx].length; i++) {
            ctx.lineTo(trails[idx][i].x, trails[idx][i].y)
          }

          ctx.strokeStyle = colors[idx % colors.length]
          ctx.lineWidth = 2
          ctx.stroke()
        }
      })

      requestAnimationFrame(animate)
    }

    // Start animation
    animate()

    // Cleanup
    return () => {
      window.removeEventListener("resize", resizeCanvas)
    }
  }, [])

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full -z-10 opacity-30" />
}
