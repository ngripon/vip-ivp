import type { ReactNode } from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import React, { useEffect, useState, useRef } from "react";
import CodeBlock from "@theme/CodeBlock";
import { Line } from "react-chartjs-2";
import "chart.js/auto";
import { ChaoticBackground } from "../components/chaotic_background";
import { ArrowRight } from "lucide-react";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  return (
    <header className="relative flex flex-col mx-auto text-center w-full py-32 overflow-hidden">
      <ChaoticBackground className="bg-transparent dark:bg-transparent" />
      <div className="relative z-10">
        <Heading
          as="h1"
          className="text-6xl font-extrabold tracking-tight leading-tight text-gray-900 dark:text-white"
        >
          {siteConfig.title}
        </Heading>
        <p className="text-xl mt-4 text-gray-700 dark:text-gray-400">
          {siteConfig.tagline}
        </p>
        <div className="mt-8 flex items-center justify-center gap-8">
          <Link
            className="bg-blue-600 text-white! py-3 px-6 rounded-lg text-lg font-medium shadow-lg hover:bg-blue-800 transition-all
            dark:bg-blue-500 dark:hover:bg-blue-700 flex items-center gap-2"
            to="/docs/overview"
          >
            Get Started
            <ArrowRight />
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Interactive bouncing ball simulation and documentation overview"
    >
      <main className="flex flex-col bg-grid-pattern dark:bg-grid-pattern-dark">
        <HomepageHeader />
        <ProsSection />
        <CodeExample />
        <CTASection />
      </main>
    </Layout>
  );
}

function ProsSection(): ReactNode {
  return (
    <section className="py-8 bg-gray-50 dark:bg-gray-900 border-t border-b">
      <div className="container mx-auto px-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow">
            <h3 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
              🌿 A natural language for dynamic systems
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Model continuous-time systems in pure Python, using simple,
              composable expressions that reflect the system's true structure.
            </p>
          </div>
          <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow">
            <h3 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            🎛 Hybrid dynamics made simple
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
            Seamlessly combine differential equations with event-driven behavior, resets, and termination logic — all in code.
            </p>
          </div>
          <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow">
            <h3 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            🧠 For thinkers, builders, and researchers
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
            Whether you're exploring ideas or publishing results, vip-ivp offers a lightweight, reliable foundation for simulation in Python.
            </p>
          </div>
          <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow">
            <h3 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            🛤️ Designed with extensibility in mind
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
            From hybrid logic to multiphysics modeling, the roadmap includes powerful abstractions like state machines, bond graphs, and discrete-time signals.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

const dt = 0.005;

function CodeExample(): ReactNode {
  const [k, setk] = useState(0.7);
  const [vMin, setVMin] = useState(0.01);
  const [data, setData] = useState<{ time: number[]; height: number[] }>({
    time: [],
    height: [],
  });

  useEffect(() => {
    // Simulate the bouncing ball system
    const simulate = () => {
      const time = [];
      const height = [];
      let t = 0;
      let h = 1;
      let v = 0;
      const g = -9.81;

      while (t <= 20) {
        time.push(t);
        height.push(h);

        v += g * dt;
        h += v * dt;

        if (h <= 0 && Math.abs(v) > vMin) {
          v = -k * v;
          h = 0;
        } else if (h <= 0) {
          break;
        }

        t += dt;
      }

      setData({ time, height });
    };

    simulate();
  }, [k, vMin]);

  return (
    <div className="m-8 p-6 rounded-lg bg-white border shadow dark:bg-gray-800 dark:border-gray-700">
      <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-4">
        Interactive Bouncing Ball Simulation
      </h2>
      <p className="text-gray-600 dark:text-gray-400 mb-6">
        Adjust the bouncing coefficient and minimum velocity using the sliders
        below to see how they affect the motion of the ball. The Python code and
        the corresponding plot are updated dynamically.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-4 rounded-lg">
          <div className="mb-2">
            <label
              htmlFor="k"
              className="block text-sm font-medium text-gray-700 dark:text-gray-400"
            >
              Bouncing Coefficient (k)
            </label>
            <input
              id="k"
              type="range"
              min={0}
              max={1.5}
              step={0.01}
              value={k}
              onChange={(e) => setk(parseFloat(e.target.value))}
              className="w-full mt-2"
            />
          </div>
          <div className="mb-2">
            <label
              htmlFor="vMin"
              className="block text-sm font-medium text-gray-700 dark:text-gray-400"
            >
              Minimum Velocity (v_min)
            </label>
            <input
              id="vMin"
              type="range"
              min={0.01}
              max={5}
              step={0.01}
              value={vMin}
              onChange={(e) => setVMin(parseFloat(e.target.value))}
              className="w-full mt-2"
            />
          </div>
          <CodeBlock language="python" className="text-gray-800">
            {`k = ${k}  # Bouncing coefficient
v_min = ${vMin}  # Minimum velocity need to bounce

# Create the system
acceleration = vip.temporal(-9.81)
velocity = vip.integrate(acceleration, x0=0)
height = vip.integrate(velocity, x0=1)

# Create the bouncing event
hit_ground = height.crosses(0, "falling")
velocity.reset_on(hit_ground, -0.8 * velocity)
vip.terminate_on(hit_ground & (abs(velocity) <= v_min))

# Solve the system
vip.solve(20, time_step=${dt})`}
          </CodeBlock>
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">
            Simulation Results
          </h3>
          <Line
            data={{
              labels: data.time,
              datasets: [
                {
                  label: "Height (m)",
                  data: data.height,
                  borderColor: "rgba(75, 192, 192, 1)",
                  backgroundColor: "rgba(75, 192, 192, 0.2)",
                  fill: true,
                },
              ],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  display: true,
                  position: "top",
                },
              },
              scales: {
                x: {
                  title: {
                    display: true,
                    text: "Time (s)",
                  },
                  ticks: {
                    callback: (value, index) =>
                      (parseFloat(value.toString()) * dt).toFixed(2),
                  },
                },
                y: {
                  title: {
                    display: true,
                    text: "Height (m)",
                  },
                  ticks: {},
                },
              },
            }}
            style={{ maxHeight: "500px" }}
          />
        </div>
      </div>
    </div>
  );
}

function CTASection(): ReactNode {
  return (
    <section className="py-16 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-center">
      <div className="container mx-auto px-8">
        <h2 className="text-3xl font-bold mb-4">
          Ready to Get Started?
        </h2>
        <p className="text-lg mb-8">
          Explore our documentation and start building with vip-ivp today.
        </p>
        <Link
          to="/docs/overview"
          className="bg-blue-600 text-white! py-3 px-6 rounded-lg text-lg font-medium shadow-lg hover:bg-blue-700 transition-all"
        >
          Get Started
        </Link>
      </div>
    </section>
  );
}
