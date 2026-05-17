/* ──────────────────────────────────────────────────────────────
   Pipeline diagram — faithful port of the team's "Pipeline
   Diagram.html". Renders at native 2200 px and scales down to
   whatever container width is supplied. Includes the bypass
   dashed arrow under the flow and the cluster-feedback curve.
   ────────────────────────────────────────────────────────────── */

function PipelineDiagram({ containerWidth = 970, dark = false }) {
  const BASE_W = 2200;
  const scale = containerWidth / BASE_W;
  // Measured native height of the diagram (title + pipeline + bypass + legend + footer).
  const BASE_H = 470;
  const scaledH = BASE_H * scale;

  return (
    <div
      style={{
        width: containerWidth,
        height: scaledH,
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: BASE_W,
          transform: `scale(${scale})`,
          transformOrigin: "top left",
          color: "#0f1722",
          fontFamily: "'Inter', sans-serif",
        }}
      >
        <PipelineFrame dark={dark} />
      </div>
    </div>
  );
}

function PipelineFrame({ dark }) {
  const C = {
    ink: "#0f1722",
    inkSoft: "#334155",
    card: "#ffffff",
    groupStroke: "#1f2937",
    input:  { bg: "#6aa5d8", ink: "#0b2a44" },
    struct: { bg: "#1c8dff", ink: "#ffffff" },
    gsplat: { bg: "#f59f2a", ink: "#3a2200" },
    anom:   { bg: "#d12947", ink: "#ffffff" },
    output: { bg: "#2ea24a", ink: "#ffffff" },
    viewer: { bg: "#0e1a2b", ink: "#ffffff" },
    bypass: "#1c8dff",
    pageBg: dark ? "#15151a" : "#f7f8fa",
  };

  return (
    <div style={{ width: 2200, padding: "0 0 0 0" }}>
      {/* TITLE ROW */}
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          margin: "0 6px 18px",
          gap: 16,
        }}
      >
        <h3
          style={{
            fontSize: 18,
            fontWeight: 600,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            margin: 0,
            color: C.ink,
          }}
        >
          Anomaly Detection Pipeline · Gaussian Splat Scenes
        </h3>
        <span
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
            color: C.inkSoft,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
          }}
        >
          v1 · process flow
        </span>
      </div>

      {/* PIPELINE FLEX */}
      <div style={{ display: "flex", alignItems: "stretch", gap: 0 }}>
        <Group label="Input" num="01" stroke={C.groupStroke} card={C.card} flex="0 0 180px" stackInner>
          <Node {...C.input} h={46}>Camera Poses</Node>
          <Node {...C.input} h={46}>Gaussian Splat Model</Node>
          <Node {...C.input} h={46}>Sparse Reconstruction</Node>
        </Group>

        <Gutter />

        <Group label="Structure Isolation" num="02" stroke={C.groupStroke} card={C.card} flex="1 1 0">
          <Node {...C.struct}>Trajectory<br/>Extraction<br/>&amp; Smoothing</Node>
          <Node {...C.struct}>PCA<br/>Alignment</Node>
          <Node {...C.struct}>Ellipsoid<br/>Coarse Filter</Node>
          <Node {...C.struct}>Ground<br/>Removal<br/>(RANSAC)</Node>
          <Node {...C.struct}>SDF Fine<br/>Filter</Node>
          <Node {...C.struct}>Statistical<br/>Outlier<br/>Removal</Node>
        </Group>

        <Gutter />

        <Group label="GSplat Extraction" num="03" stroke={C.groupStroke} card={C.card} flex="0 0 220px">
          <Node {...C.gsplat}>Sphere<br/>Filter</Node>
          <Node {...C.gsplat}>Scale<br/>Filter</Node>
        </Group>

        <Gutter />

        <Group label="Anomaly Detection" num="04" stroke={C.groupStroke} card={C.card} flex="1 1 0">
          <Node {...C.anom}>Feature<br/>Extraction</Node>
          <Node {...C.anom}>Feature<br/>Normalisation</Node>
          <Node {...C.anom}>Randomised<br/>RX Detector</Node>
          <Node {...C.anom}>CFAR<br/>Thresholding</Node>
          <Node {...C.anom}>DBSCAN<br/>Clustering</Node>
        </Group>

        <Gutter />

        <Group label="Output" num="05" stroke={C.groupStroke} card={C.card} flex="0 0 180px" stackInner stackGap={14}>
          <Node {...C.output} h={70}>Filtered<br/>Point Cloud</Node>
          <Node {...C.output} h={70}>Anomaly<br/>Point Cloud</Node>
        </Group>

        <Gutter />

        {/* VIEWER */}
        <div style={{ flex: "0 0 250px" }}>
          <div
            style={{
              borderRadius: 10,
              background: C.viewer.bg,
              color: C.viewer.ink,
              padding: "18px 16px",
              display: "flex",
              flexDirection: "column",
              gap: 6,
              minHeight: 230,
              justifyContent: "center",
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>
              Decision-Support Viewer
            </div>
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              <li style={{ fontSize: 11.5, lineHeight: 1.6, color: "#d6dde8" }}>
                Real-time 3D rendering
              </li>
              <li style={{ fontSize: 11.5, lineHeight: 1.6, color: "#d6dde8" }}>
                Anomaly cluster navigation
              </li>
              <li style={{ fontSize: 11.5, lineHeight: 1.6, color: "#d6dde8" }}>
                Bounding-box overlays
              </li>
              <li style={{ fontSize: 11.5, lineHeight: 1.6, color: "#d6dde8" }}>
                Inspector review &amp; action
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* BYPASS + FEEDBACK FLOW */}
      <div style={{ position: "relative", marginTop: 22, height: 60 }}>
        <svg
          viewBox="0 0 2200 60"
          preserveAspectRatio="none"
          style={{ width: "100%", height: "100%", display: "block", overflow: "visible" }}
          aria-hidden="true"
        >
          <defs>
            <marker id="arr-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" fill={C.bypass} />
            </marker>
            <marker id="arr-dark" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" fill={C.viewer.bg} />
            </marker>
          </defs>
          {/* Sparse Reconstruction → GSplat Extraction (bypass Stage 1) */}
          <path
            d="M 90 5 L 90 35 L 956 35 L 956 5"
            fill="none"
            stroke={C.bypass}
            strokeWidth={2}
            strokeDasharray="6 6"
            markerEnd="url(#arr-blue)"
          />
          {/* DBSCAN clustering → Decision-Support Viewer (feedback) */}
          <path
            d="M 1640 5 L 1640 45 L 2075 45 L 2075 10"
            fill="none"
            stroke={C.viewer.bg}
            strokeWidth={2}
            markerEnd="url(#arr-dark)"
          />
        </svg>
        <div
          style={{
            position: "absolute",
            left: "24%",
            top: 28,
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10.5,
            letterSpacing: "0.08em",
            color: C.bypass,
            textTransform: "uppercase",
            background: C.pageBg,
            padding: "0 6px",
          }}
        >
          Bypass Stage 1
        </div>
        <div
          style={{
            position: "absolute",
            left: "82%",
            top: 28,
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10.5,
            letterSpacing: "0.08em",
            color: C.viewer.bg,
            textTransform: "uppercase",
            background: C.pageBg,
            padding: "0 6px",
          }}
        >
          Cluster feedback
        </div>
      </div>

      {/* LEGEND */}
      <div
        style={{
          marginTop: 18,
          display: "flex",
          flexWrap: "wrap",
          gap: "12px 28px",
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 11,
          color: C.inkSoft,
          letterSpacing: "0.04em",
        }}
      >
        <Legend swatch={C.input.bg}>Input data</Legend>
        <Legend swatch={C.struct.bg}>Structure Isolation</Legend>
        <Legend swatch={C.gsplat.bg}>GSplat Extraction</Legend>
        <Legend swatch={C.anom.bg}>Anomaly Detection</Legend>
        <Legend swatch={C.output.bg}>Output</Legend>
        <LegendLine solid color="#1f2937">Primary flow</LegendLine>
        <LegendLine color={C.bypass}>Bypass / Feedback</LegendLine>
      </div>

      {/* FOOTER */}
      <div
        style={{
          marginTop: 20,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10.5,
          color: C.inkSoft,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
        }}
      >
        <span></span>
        <span></span>
      </div>
    </div>
  );
}

/* GROUP container with corner labels and inner node row -------- */
function Group({ label, num, stroke, card, flex, stackInner = false, stackGap = 10, children }) {
  return (
    <section
      style={{
        flex,
        position: "relative",
        border: `1.5px solid ${stroke}`,
        borderRadius: 14,
        background: card,
        padding: "30px 14px 16px",
        minHeight: 230,
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        minWidth: 0,
      }}
    >
      <span
        style={{
          position: "absolute",
          top: -10,
          left: 16,
          background: card,
          padding: "0 10px",
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "#0f1722",
        }}
      >
        {label}
      </span>
      <span
        style={{
          position: "absolute",
          top: -10,
          right: 16,
          background: card,
          padding: "0 10px",
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10,
          letterSpacing: "0.08em",
          color: "#334155",
        }}
      >
        {num}
      </span>
      <div
        style={{
          display: "flex",
          flexDirection: stackInner ? "column" : "row",
          gap: stackInner ? stackGap : 10,
          alignItems: "stretch",
          justifyContent: "center",
          minWidth: 0,
        }}
      >
        {children}
      </div>
    </section>
  );
}

/* NODE — colored pill -------------------------------------- */
function Node({ bg, ink, h, children }) {
  return (
    <div
      style={{
        flex: "1 1 0",
        minWidth: 0,
        borderRadius: 8,
        padding: h ? "12px 10px" : "14px 8px",
        fontSize: 13,
        fontWeight: 600,
        lineHeight: 1.3,
        textAlign: "center",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        minHeight: h || 90,
        background: bg,
        color: ink,
        boxShadow: "0 1px 0 rgba(15,23,42,0.04)",
      }}
    >
      <span>{children}</span>
    </div>
  );
}

/* GUTTER — black arrow between groups ---------------------- */
function Gutter() {
  return (
    <div style={{ flex: "0 0 38px", position: "relative" }}>
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: 0,
          right: 12,
          borderTop: "2px solid #111827",
          transform: "translateY(-1px)",
        }}
      />
      <div
        style={{
          position: "absolute",
          top: "50%",
          right: 8,
          width: 0,
          height: 0,
          borderLeft: "10px solid #111827",
          borderTop: "6px solid transparent",
          borderBottom: "6px solid transparent",
          transform: "translateY(-50%)",
        }}
      />
    </div>
  );
}

function Legend({ swatch, children }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
      <span style={{ width: 14, height: 14, borderRadius: 3, background: swatch, display: "inline-block" }} />
      {children}
    </span>
  );
}
function LegendLine({ solid, color, children }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
      <span
        style={{
          width: 26,
          height: 0,
          borderTop: `2px ${solid ? "solid" : "dashed"} ${color}`,
        }}
      />
      {children}
    </span>
  );
}

window.PipelineDiagram = PipelineDiagram;
