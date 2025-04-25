import type { ReactNode } from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import React, { useEffect, useState } from "react";
import CodeBlock from "@theme/CodeBlock";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className="hero bg-primary text-white py-8">
      <div className="container mx-auto text-center">
        <Heading as="h1" className="text-4xl font-bold">
          {siteConfig.title}
        </Heading>
        <p className="text-lg mt-4">{siteConfig.tagline}</p>
        <div className="mt-6">
          <Link
            className="bg-secondary text-white py-2 px-4 rounded-lg text-lg"
            to="/docs/intro"
          >
            Learn
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
      description="Description will go into a meta tag in <head />"
    >
      <HomepageHeader />
      <main>
        <CodeExample />
      </main>
    </Layout>
  );
}

function CodeExample(): ReactNode {
  const [k, setk] = useState(0.7);

  return (
    <div className="m-8">
      <label htmlFor="k" className="block text-sm font-medium text-gray-700">
        Bouncing coefficient
      </label>
      <input
        id="k"
        type="range"
        min={0}
        max={1.5}
        step={0.01}
        value={k}
        onChange={(e) => setk(e.target.value)}
        className="w-full mt-2"
      />
      <CodeBlock language="python" className="mt-4">
        {`k = ${k}  # Bouncing coefficient
v_min = 0.01  # Minimum velocity need to bounce

# Create the system
acceleration = vip.temporal(-9.81)
velocity = vip.integrate(acceleration, x0=0)
height = vip.integrate(velocity, x0=1)

# Create the bouncing event
bounce = vip.where(abs(velocity) > v_min, velocity.action_reset_to(-k * velocity), vip.action_terminate)
height.on_crossing(0, bounce, terminal=False, direction="falling")

# Solve the system
vip.solve(20, time_step=0.001)`}
      </CodeBlock>
    </div>
  );
}
