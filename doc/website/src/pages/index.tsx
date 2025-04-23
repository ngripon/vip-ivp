import type { ReactNode } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import React, { useEffect, useState } from "react";
import CodeBlock from "@theme/CodeBlock";

import styles from "./index.module.css";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx("hero hero--primary", styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
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
        <CodeExample/>
      </main>
    </Layout>
  );
}

function CodeExample(): ReactNode {
  const [k, setk]=useState(0.7)

  return (
    <div>
      <label htmlFor="k">Bouncing coefficient</label>
      <input id="k" type="range" min={0} max={1.5} step={0.01} value={k} onChange={(e)=>setk(e.target.value)} />
      <CodeBlock language="python">
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
