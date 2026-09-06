import { useMemo, useState } from "react";
import {
  BarChart3,
  Brain,
  CheckCircle2,
  ChevronRight,
  Database,
  Download,
  FileArchive,
  FileText,
  Play,
  UploadCloud
} from "lucide-react";

type LogLine = {
  stage: string;
  message: string;
  detail?: string;
  level?: string;
};

type Artifact = {
  name: string;
  kind: string;
  path: string;
  downloadName: string;
};

type RunPayload = {
  runId: string;
  modelName: string;
  datasetName: string;
  profile: Record<string, any>;
  preprocessing: Record<string, any>;
  metrics: Record<string, any>;
  anomalyResult: {
    totalRows: number;
    totalAnomalies: number;
    anomalyRatio: number;
    thresholdUsed: number;
    anomalies: Array<Record<string, any>>;
    conceptDriftEvents: Array<{ rowIndexRange: number[]; description: string }>;
  };
  pipelineLogs: LogLine[];
  comparison: Array<Record<string, unknown>>;
  artifacts: Artifact[];
};

type PageKey = "upload" | "briefing" | "results" | "downloads";

const pendingSteps = [
  "Dataset received",
  "Profile checked",
  "Values prepared",
  "Normal pattern learned",
  "Change controller trained",
  "Memory and adapter prepared",
  "Threshold tuned",
  "Explanations created",
  "Download files saved"
];

const pages: Array<{ key: PageKey; label: string; icon: React.ReactNode }> = [
  { key: "upload", label: "Upload", icon: <UploadCloud size={18} /> },
  { key: "briefing", label: "Execution Briefing", icon: <Brain size={18} /> },
  { key: "results", label: "Results", icon: <BarChart3 size={18} /> },
  { key: "downloads", label: "Downloads", icon: <Download size={18} /> }
];

export default function App() {
  const [activePage, setActivePage] = useState<PageKey>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [modelName, setModelName] = useState("DyMETER");
  const [run, setRun] = useState<RunPayload | null>(null);
  const [steps, setSteps] = useState<LogLine[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const severityCounts = useMemo(() => {
    const counts: Record<string, number> = {
      Normal: Math.max((run?.anomalyResult.totalRows ?? 0) - (run?.anomalyResult.totalAnomalies ?? 0), 0),
      Warning: 0,
      Critical: 0,
      Emergency: 0
    };
    run?.anomalyResult.anomalies.forEach((item) => {
      const key = `${item.severityLevel ?? "Warning"}`;
      counts[key] = (counts[key] ?? 0) + 1;
    });
    return counts;
  }, [run]);

  async function startRun() {
    if (!file) return;
    setLoading(true);
    setError("");
    setRun(null);
    setActivePage("briefing");
    setSteps(pendingSteps.map((stage) => ({ stage, message: "Waiting for this step to finish.", detail: "" })));

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("modelName", modelName);
      formData.append("config", JSON.stringify({ forceNumpy: false }));
      const response = await fetch("/api/runs", { method: "POST", body: formData });
      if (!response.ok) {
        throw new Error(`Run failed with status ${response.status}`);
      }
      const payload = (await response.json()) as RunPayload;
      setRun(payload);
      setSteps(payload.pipelineLogs);
      setActivePage("results");
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "The run could not be completed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="appShell">
      <aside className="sideNav">
        <div className="brandBlock">
          <div className="brandMark">ED</div>
          <div>
            <h1>Enhanced Dymeter</h1>
            <p>Adaptive anomaly awareness</p>
          </div>
        </div>
        <nav>
          {pages.map((page) => (
            <button
              key={page.key}
              className={activePage === page.key ? "navButton active" : "navButton"}
              onClick={() => setActivePage(page.key)}
            >
              {page.icon}
              <span>{page.label}</span>
            </button>
          ))}
        </nav>
        <div className="runBadge">
          <span>Current run</span>
          <strong>{run?.runId ?? "Not started"}</strong>
        </div>
      </aside>

      <section className="pageSurface">
        {activePage === "upload" && (
          <UploadPage
            file={file}
            modelName={modelName}
            loading={loading}
            error={error}
            onFileChange={setFile}
            onModelChange={setModelName}
            onStart={startRun}
          />
        )}
        {activePage === "briefing" && <BriefingPage steps={steps} loading={loading} error={error} />}
        {activePage === "results" && <ResultsPage run={run} severityCounts={severityCounts} />}
        {activePage === "downloads" && <DownloadsPage run={run} />}
      </section>
    </main>
  );
}

function UploadPage({
  file,
  modelName,
  loading,
  error,
  onFileChange,
  onModelChange,
  onStart
}: {
  file: File | null;
  modelName: string;
  loading: boolean;
  error: string;
  onFileChange: (file: File | null) => void;
  onModelChange: (modelName: string) => void;
  onStart: () => void;
}) {
  return (
    <div className="pageGrid uploadPage">
      <div className="pageIntro">
        <p className="eyebrow">Start</p>
        <h2>Train a clear, adaptive run.</h2>
        <p>
          Upload a dataset, choose the execution path, and let the app produce decisions, explanations,
          downloadable visuals, and a report.
        </p>
      </div>

      <div className="largePanel">
        <label className="fieldLabel" htmlFor="dataset">
          Dataset
        </label>
        <input
          id="dataset"
          className="fileInput"
          type="file"
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
        />
        <div className="selectedFile">
          <Database size={22} />
          <span>{file?.name ?? "No dataset selected"}</span>
        </div>

        <label className="fieldLabel" htmlFor="model">
          Model
        </label>
        <select id="model" className="selectInput" value={modelName} onChange={(event) => onModelChange(event.target.value)}>
          <option value="DyMETER">DyMETER</option>
          <option value="METER">METER</option>
          <option value="D3R">D3R</option>
          <option value="SARAD">SARAD</option>
          <option value="EnhancedDymeterMANN">Enhanced Dymeter MANN</option>
        </select>

        <button className="primaryButton" disabled={!file || loading} onClick={onStart}>
          <Play size={18} />
          {loading ? "Training" : "Train"}
        </button>
        {error && <p className="errorText">{error}</p>}
      </div>
    </div>
  );
}

function BriefingPage({ steps, loading, error }: { steps: LogLine[]; loading: boolean; error: string }) {
  const visibleSteps = steps.length ? steps : pendingSteps.map((stage) => ({ stage, message: "Run has not started.", detail: "" }));
  return (
    <div className="pageStack">
      <PageHeader
        eyebrow="Execution Briefing"
        title={loading ? "Training is running." : "Training steps"}
        copy="Each completed step is explained in plain language so the process is easy to follow."
      />
      {error && <div className="alertBox">{error}</div>}
      <div className="stepList">
        {visibleSteps.map((step, index) => (
          <article className="stepCard" key={`${step.stage}-${index}`}>
            <div className="stepIcon">{loading && index === 0 ? <Brain size={22} /> : <CheckCircle2 size={22} />}</div>
            <div>
              <h3>{step.stage}</h3>
              <p>{step.message}</p>
              {step.detail && <span>{step.detail}</span>}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

function ResultsPage({ run, severityCounts }: { run: RunPayload | null; severityCounts: Record<string, number> }) {
  if (!run) {
    return <EmptyPage icon={<BarChart3 size={40} />} title="No results yet" copy="Train a dataset first, then return here." />;
  }
  const metricSuffix = run.metrics.metricMode === "unlabeledProxy" ? " (proxy)" : "";

  return (
    <div className="pageStack">
      <PageHeader
        eyebrow="Results"
        title={`${displayModelName(run.modelName)} completed`}
        copy="The application keeps visual files downloadable and shows the main decisions here."
      />

      <div className="metricGrid">
        <Metric label="Rows" value={run.anomalyResult.totalRows} />
        <Metric label="Flagged Rows" value={run.anomalyResult.totalAnomalies} />
        <Metric label="Anomaly Ratio" value={formatMetric(run.anomalyResult.anomalyRatio)} />
        <Metric label="Threshold" value={formatMetric(run.anomalyResult.thresholdUsed)} />
        <Metric label={`F1 Score${metricSuffix}`} value={formatMetric(run.metrics.f1Score)} />
        <Metric label={`AUC ROC${metricSuffix}`} value={formatMetric(run.metrics.aucRoc)} />
      </div>

      <div className="splitGrid">
        <section className="largePanel">
          <h3>Severity Summary</h3>
          <div className="severityRows">
            {Object.entries(severityCounts).map(([label, count]) => (
              <div className="severityRow" key={label}>
                <span>{label}</span>
                <strong>{count}</strong>
              </div>
            ))}
          </div>
        </section>

        <section className="largePanel">
          <h3>Changed Behavior Windows</h3>
          <div className="plainList">
            {run.anomalyResult.conceptDriftEvents.length ? (
              run.anomalyResult.conceptDriftEvents.slice(0, 8).map((event, index) => (
                <p key={index}>
                  Rows {event.rowIndexRange[0]} to {event.rowIndexRange[1]}: {event.description}
                </p>
              ))
            ) : (
              <p>No changed behavior window was detected.</p>
            )}
          </div>
        </section>
      </div>

      <section className="largePanel">
        <h3>Top Flagged Rows</h3>
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                <th>Row</th>
                <th>Score</th>
                <th>Route</th>
                <th>Severity</th>
                <th>Explanation</th>
              </tr>
            </thead>
            <tbody>
              {run.anomalyResult.anomalies.slice(0, 30).map((item) => (
                <tr key={item.rowIndex}>
                  <td>{item.rowIndex}</td>
                  <td>{formatMetric(item.anomalyScore)}</td>
                  <td>{item.detectorUsed}</td>
                  <td>{item.severityLevel}</td>
                  <td>{item.naturalLanguageExplanation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function DownloadsPage({ run }: { run: RunPayload | null }) {
  if (!run) {
    return <EmptyPage icon={<Download size={40} />} title="No downloads yet" copy="Train a dataset to create reports and image files." />;
  }

  const grouped = {
    package: run.artifacts.filter((item) => item.kind === "zip" || item.kind === "pdf" || item.kind === "html" || item.kind === "json"),
    images: run.artifacts.filter((item) => item.kind === "png")
  };

  return (
    <div className="pageStack">
      <PageHeader
        eyebrow="Downloads"
        title="Files created for this run"
        copy="Visualizations are saved as files only. They are not embedded in the application."
      />

      <div className="downloadGroup">
        {grouped.package.map((artifact) => (
          <DownloadCard key={artifact.path} runId={run.runId} artifact={artifact} featured />
        ))}
      </div>

      <h3 className="subTitle">Image files</h3>
      <div className="downloadGrid">
        {grouped.images.map((artifact) => (
          <DownloadCard key={artifact.path} runId={run.runId} artifact={artifact} />
        ))}
      </div>
    </div>
  );
}

function DownloadCard({ runId, artifact, featured = false }: { runId: string; artifact: Artifact; featured?: boolean }) {
  const href = `/api/runs/${runId}/artifacts/${artifact.path}`;
  const Icon = artifact.kind === "zip" ? FileArchive : artifact.kind === "png" ? BarChart3 : FileText;
  return (
    <a className={featured ? "downloadCard featured" : "downloadCard"} href={href} download={artifact.downloadName}>
      <Icon size={24} />
      <span>
        <strong>{artifact.name}</strong>
        <small>{artifact.downloadName}</small>
      </span>
      <ChevronRight size={18} />
    </a>
  );
}

function PageHeader({ eyebrow, title, copy }: { eyebrow: string; title: string; copy: string }) {
  return (
    <header className="pageHeader">
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      <p>{copy}</p>
    </header>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metricCard">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyPage({ icon, title, copy }: { icon: React.ReactNode; title: string; copy: string }) {
  return (
    <div className="emptyPage">
      <div className="emptyIcon">{icon}</div>
      <h2>{title}</h2>
      <p>{copy}</p>
    </div>
  );
}

function displayModelName(modelName: string) {
  if (modelName === "EnhancedDymeterMANN") return "Enhanced Dymeter MANN";
  return modelName;
}

function formatMetric(value: unknown) {
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(4) : "n/a";
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(4) : `${value}`;
}
