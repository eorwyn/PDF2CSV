import { useEffect, useMemo, useRef, useState } from "react";
import { listModels } from "./lib/backend";
import { runExtraction } from "./lib/extractor";
import { downloadCsv, downloadXlsx } from "./lib/exporters";
import {
  DEFAULT_PROMPT_CONFIG,
  parsePromptConfigMarkdown,
  promptConfigToMarkdown,
} from "./lib/prompts";
import {
  DEFAULT_OLLAMA_SETTINGS,
  DEFAULT_QUALITY_SETTINGS,
  sanitizeOllamaSettings,
  sanitizeQualitySettings,
} from "./lib/settings";
import type {
  BackendKind,
  ExtractionQualitySettings,
  ExtractionRow,
  LogEntry,
  LogLevel,
  OllamaGenerationSettings,
  PromptConfig,
  RunProgress,
} from "./types";

interface SavedSettings {
  rememberSettings: boolean;
  backendKind: BackendKind;
  baseUrl: string;
  selectedModel: string;
  manualModel: string;
  concurrency: number;
  retries: number;
  promptConfig: PromptConfig;
  qualitySettings: ExtractionQualitySettings;
  ollamaSettings: OllamaGenerationSettings;
}

const SETTINGS_KEY = "pdf2csv.settings.v1";

function createLog(level: LogLevel, message: string): LogEntry {
  const id =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random()}`;
  return {
    id,
    level,
    message,
    timestamp: new Date().toLocaleTimeString(),
  };
}

function normalizePdfList(inputFiles: File[]): File[] {
  const map = new Map<string, File>();
  for (const file of inputFiles) {
    if (!file.name.toLowerCase().endsWith(".pdf")) continue;
    const key = `${file.name}:${file.size}:${file.lastModified}`;
    map.set(key, file);
  }
  return Array.from(map.values());
}

function mergeFiles(existing: File[], incoming: File[]): File[] {
  return normalizePdfList([...existing, ...incoming]);
}

function buildDownloadName(extension: "csv" | "xlsx"): string {
  const stamp = new Date().toISOString().replace(/[:]/g, "-");
  return `pdf-paragraphs-${stamp}.${extension}`;
}

function triggerTextDownload(text: string, fileName: string): void {
  const blob = new Blob([text], { type: "text/markdown;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function toNumber(value: string, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return parsed;
}

export default function App(): JSX.Element {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const promptFileInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [backendKind, setBackendKind] = useState<BackendKind>("openai");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [manualModel, setManualModel] = useState("");
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState("");

  const [files, setFiles] = useState<File[]>([]);
  const [isDragActive, setIsDragActive] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [concurrency, setConcurrency] = useState(2);
  const [retries, setRetries] = useState(2);
  const [rememberSettings, setRememberSettings] = useState(false);
  const [promptConfig, setPromptConfig] =
    useState<PromptConfig>(DEFAULT_PROMPT_CONFIG);
  const [qualitySettings, setQualitySettings] = useState<ExtractionQualitySettings>(
    DEFAULT_QUALITY_SETTINGS,
  );
  const [ollamaSettings, setOllamaSettings] = useState<OllamaGenerationSettings>(
    DEFAULT_OLLAMA_SETTINGS,
  );

  const [rows, setRows] = useState<ExtractionRow[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [progress, setProgress] = useState<RunProgress | null>(null);

  const activeModel = useMemo(
    () => manualModel.trim() || selectedModel.trim(),
    [manualModel, selectedModel],
  );

  useEffect(() => {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as SavedSettings;
      if (!parsed.rememberSettings) return;
      setRememberSettings(true);
      setBackendKind(parsed.backendKind);
      setBaseUrl(parsed.baseUrl);
      setSelectedModel(parsed.selectedModel);
      setManualModel(parsed.manualModel);
      setConcurrency(parsed.concurrency);
      setRetries(parsed.retries);
      setQualitySettings(sanitizeQualitySettings(parsed.qualitySettings));
      setOllamaSettings(sanitizeOllamaSettings(parsed.ollamaSettings));
      if (
        parsed.promptConfig?.textFilterSystem &&
        parsed.promptConfig?.visionSystem
      ) {
        setPromptConfig(parsed.promptConfig);
      }
    } catch {
      // Ignore malformed local settings.
    }
  }, []);

  useEffect(() => {
    if (!rememberSettings) {
      localStorage.removeItem(SETTINGS_KEY);
      return;
    }

    const payload: SavedSettings = {
      rememberSettings,
      backendKind,
      baseUrl,
      selectedModel,
      manualModel,
      concurrency,
      retries,
      promptConfig,
      qualitySettings,
      ollamaSettings,
    };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(payload));
  }, [
    rememberSettings,
    backendKind,
    baseUrl,
    selectedModel,
    manualModel,
    concurrency,
    retries,
    promptConfig,
    qualitySettings,
    ollamaSettings,
  ]);

  function appendLog(level: LogLevel, message: string): void {
    setLogs((previous) => [...previous, createLog(level, message)]);
  }

  function onChooseFiles(filesLike: FileList | null): void {
    if (!filesLike) return;
    const incoming = Array.from(filesLike);
    const merged = mergeFiles(files, incoming);
    const added = merged.length - files.length;
    setFiles(merged);
    if (added > 0) {
      appendLog("info", `Added ${added} PDF file(s).`);
    }
  }

  function removeFile(fileToRemove: File): void {
    setFiles((previous) => {
      const next = previous.filter(
        (file) =>
          !(
            file.name === fileToRemove.name &&
            file.size === fileToRemove.size &&
            file.lastModified === fileToRemove.lastModified
          ),
      );
      return next;
    });
  }

  async function handleLoadModels(): Promise<void> {
    setModelError("");
    setModelLoading(true);
    try {
      const loaded = await listModels({
        kind: backendKind,
        baseUrl,
        apiKey,
      });
      setModels(loaded);
      if (!selectedModel || !loaded.includes(selectedModel)) {
        setSelectedModel(loaded[0]);
      }
      appendLog("info", `Loaded ${loaded.length} model(s) from endpoint.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setModelError(message);
      setModels([]);
      appendLog(
        "warning",
        `Model listing failed. You can still enter a manual model ID. ${message}`,
      );
    } finally {
      setModelLoading(false);
    }
  }

  async function handleRun(): Promise<void> {
    if (!baseUrl.trim()) {
      appendLog("error", "Base URL is required.");
      return;
    }
    if (!activeModel) {
      appendLog(
        "error",
        "Select a model from the dropdown or provide a manual model ID.",
      );
      return;
    }
    if (files.length === 0) {
      appendLog("error", "Add at least one PDF before running extraction.");
      return;
    }

    setRows([]);
    setLogs([]);
    setProgress({
      totalPdfs: files.length,
      completedPdfs: 0,
      currentPdf: "",
      currentPage: 0,
      totalPagesForCurrent: 0,
    });
    setIsRunning(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const extractionRows = await runExtraction(files, {
        config: {
          kind: backendKind,
          baseUrl: baseUrl.trim(),
          apiKey,
          model: activeModel,
          ollama: backendKind === "ollama" ? ollamaSettings : undefined,
        },
        prompts: promptConfig,
        quality: qualitySettings,
        fileConcurrency: concurrency,
        retries,
        signal: controller.signal,
        onLog: appendLog,
        onProgress: setProgress,
      });
      setRows(extractionRows);
      appendLog(
        "info",
        `Completed. Final dataset contains ${extractionRows.length} row(s).`,
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        appendLog("warning", "Extraction canceled by user.");
      } else {
        const message =
          error instanceof Error ? error.message : "Unknown extraction error";
        appendLog("error", `Extraction stopped with an error: ${message}`);
      }
    } finally {
      setIsRunning(false);
      abortRef.current = null;
    }
  }

  function handleCancel(): void {
    abortRef.current?.abort();
    appendLog("warning", "Cancel requested. Stopping in-flight work...");
  }

  async function handlePromptFileImport(filesLike: FileList | null): Promise<void> {
    if (!filesLike || filesLike.length === 0) return;
    try {
      const file = filesLike[0];
      const markdown = await file.text();
      const parsed = parsePromptConfigMarkdown(markdown);
      setPromptConfig(parsed);
      appendLog("info", `Loaded prompt configuration from ${file.name}.`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown prompt import error";
      appendLog("error", `Prompt import failed: ${message}`);
    }
  }

  function handlePromptDownload(): void {
    const markdown = promptConfigToMarkdown(promptConfig);
    triggerTextDownload(markdown, "pdf2csv-prompts.md");
    appendLog("info", "Downloaded prompt configuration markdown.");
  }

  const overallPercent = useMemo(() => {
    if (!progress || progress.totalPdfs === 0) return 0;
    return Math.round((progress.completedPdfs / progress.totalPdfs) * 100);
  }, [progress]);

  return (
    <div className="app-shell">
      <header className="hero">
        <h1>PDF Main-Body Paragraph Extractor</h1>
        <p>
          Upload PDFs, keep core narrative paragraphs, deduplicate exact
          repeats per file, and download CSV or XLSX.
        </p>
      </header>

      <main className="grid">
        <section className="panel">
          <h2>1) Backend Configuration</h2>

          <label className="field">
            <span>Backend type</span>
            <select
              value={backendKind}
              onChange={(event) => {
                setBackendKind(event.target.value as BackendKind);
                setModelError("");
                setModels([]);
                setSelectedModel("");
              }}
              disabled={isRunning}
            >
              <option value="openai">OpenAI-compatible</option>
              <option value="ollama">Ollama-compatible</option>
            </select>
          </label>

          <label className="field">
            <span>Base URL</span>
            <input
              value={baseUrl}
              onChange={(event) => setBaseUrl(event.target.value)}
              placeholder={
                backendKind === "openai"
                  ? "https://your-endpoint.example/v1"
                  : "http://localhost:11434"
              }
              disabled={isRunning}
            />
          </label>

          <label className="field">
            <span>API key (OpenAI-compatible only)</span>
            <input
              type="password"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              placeholder="sk-..."
              disabled={isRunning || backendKind === "ollama"}
            />
          </label>

          <div className="inline-actions">
            <button
              type="button"
              onClick={handleLoadModels}
              disabled={isRunning || modelLoading || !baseUrl.trim()}
            >
              {modelLoading ? "Loading..." : "Load Models"}
            </button>
          </div>

          <label className="field">
            <span>Endpoint model list</span>
            <select
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
              disabled={isRunning || models.length === 0}
            >
              <option value="">
                {models.length === 0 ? "No models loaded" : "Select model"}
              </option>
              {models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Manual model ID (overrides dropdown when filled)</span>
            <input
              value={manualModel}
              onChange={(event) => setManualModel(event.target.value)}
              placeholder="e.g. gpt-4o-mini or llama3.1:8b"
              disabled={isRunning}
            />
          </label>
          <p className="muted">
            Use a vision-capable model for scanned/image-only PDFs.
          </p>

          {backendKind === "ollama" && (
            <div className="subpanel">
              <h3>Ollama Advanced Settings</h3>
              <div className="split">
                <label className="field">
                  <span>Temperature</span>
                  <input
                    type="number"
                    min={0}
                    max={2}
                    step={0.05}
                    value={ollamaSettings.temperature}
                    onChange={(event) =>
                      setOllamaSettings((previous) =>
                        sanitizeOllamaSettings({
                          ...previous,
                          temperature: toNumber(
                            event.target.value,
                            previous.temperature,
                          ),
                        }),
                      )
                    }
                    disabled={isRunning}
                  />
                </label>
                <label className="field">
                  <span>Top P</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={ollamaSettings.topP}
                    onChange={(event) =>
                      setOllamaSettings((previous) =>
                        sanitizeOllamaSettings({
                          ...previous,
                          topP: toNumber(event.target.value, previous.topP),
                        }),
                      )
                    }
                    disabled={isRunning}
                  />
                </label>
                <label className="field">
                  <span>Top K</span>
                  <input
                    type="number"
                    min={0}
                    max={500}
                    step={1}
                    value={ollamaSettings.topK}
                    onChange={(event) =>
                      setOllamaSettings((previous) =>
                        sanitizeOllamaSettings({
                          ...previous,
                          topK: toNumber(event.target.value, previous.topK),
                        }),
                      )
                    }
                    disabled={isRunning}
                  />
                </label>
                <label className="field">
                  <span>Min P</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={ollamaSettings.minP}
                    onChange={(event) =>
                      setOllamaSettings((previous) =>
                        sanitizeOllamaSettings({
                          ...previous,
                          minP: toNumber(event.target.value, previous.minP),
                        }),
                      )
                    }
                    disabled={isRunning}
                  />
                </label>
                <label className="field">
                  <span>Repeat penalty</span>
                  <input
                    type="number"
                    min={0.5}
                    max={3}
                    step={0.01}
                    value={ollamaSettings.repeatPenalty}
                    onChange={(event) =>
                      setOllamaSettings((previous) =>
                        sanitizeOllamaSettings({
                          ...previous,
                          repeatPenalty: toNumber(
                            event.target.value,
                            previous.repeatPenalty,
                          ),
                        }),
                      )
                    }
                    disabled={isRunning}
                  />
                </label>
                <label className="field">
                  <span>Context size (num_ctx)</span>
                  <input
                    type="number"
                    min={256}
                    max={262144}
                    step={256}
                    value={ollamaSettings.contextSize}
                    onChange={(event) =>
                      setOllamaSettings((previous) =>
                        sanitizeOllamaSettings({
                          ...previous,
                          contextSize: toNumber(
                            event.target.value,
                            previous.contextSize,
                          ),
                        }),
                      )
                    }
                    disabled={isRunning}
                  />
                </label>
              </div>

              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={ollamaSettings.useNativeToolCalling}
                  onChange={(event) =>
                    setOllamaSettings((previous) => ({
                      ...previous,
                      useNativeToolCalling: event.target.checked,
                    }))
                  }
                  disabled={isRunning}
                />
                <span>Enable Ollama native tool calling for structured JSON</span>
              </label>

              <div className="inline-actions">
                <button
                  type="button"
                  onClick={() => setOllamaSettings(DEFAULT_OLLAMA_SETTINGS)}
                  disabled={isRunning}
                >
                  Reset Ollama Defaults
                </button>
              </div>
            </div>
          )}

          {modelError && <p className="alert warning">{modelError}</p>}

          <div className="split">
            <label className="field">
              <span>PDF concurrency</span>
              <input
                type="number"
                min={1}
                max={8}
                value={concurrency}
                onChange={(event) =>
                  setConcurrency(Math.max(1, Number(event.target.value) || 1))
                }
                disabled={isRunning}
              />
            </label>
            <label className="field">
              <span>Retries per LLM chunk</span>
              <input
                type="number"
                min={0}
                max={6}
                value={retries}
                onChange={(event) =>
                  setRetries(Math.max(0, Number(event.target.value) || 0))
                }
                disabled={isRunning}
              />
            </label>
          </div>

          <label className="checkbox">
            <input
              type="checkbox"
              checked={rememberSettings}
              onChange={(event) => setRememberSettings(event.target.checked)}
              disabled={isRunning}
            />
            <span>
              Remember settings on this browser (opt-in, excludes API key)
            </span>
          </label>
        </section>

        <section className="panel">
          <h2>2) Add PDFs and Run</h2>

          <div
            className={`dropzone ${isDragActive ? "active" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragActive(true);
            }}
            onDragLeave={(event) => {
              event.preventDefault();
              setIsDragActive(false);
            }}
            onDrop={(event) => {
              event.preventDefault();
              setIsDragActive(false);
              onChooseFiles(event.dataTransfer.files);
            }}
          >
            <p>Drag and drop one or more PDFs here.</p>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isRunning}
            >
              Browse PDFs
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              multiple
              onChange={(event) => onChooseFiles(event.target.files)}
              hidden
            />
          </div>

          <div className="file-list">
            {files.length === 0 && <p className="muted">No PDFs selected yet.</p>}
            {files.map((file) => (
              <div key={`${file.name}-${file.size}-${file.lastModified}`} className="file-item">
                <div>
                  <strong>{file.name}</strong>
                  <small>{(file.size / 1024).toFixed(1)} KB</small>
                </div>
                <button
                  type="button"
                  onClick={() => removeFile(file)}
                  disabled={isRunning}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>

          <div className="inline-actions">
            <button
              type="button"
              className="primary"
              onClick={handleRun}
              disabled={isRunning || files.length === 0 || !baseUrl.trim()}
            >
              Run Extraction
            </button>
            <button type="button" onClick={handleCancel} disabled={!isRunning}>
              Cancel
            </button>
          </div>

          <div className="progress-panel">
            <p>
              Overall: {progress?.completedPdfs ?? 0}/{progress?.totalPdfs ?? 0} PDFs (
              {overallPercent}%)
            </p>
            {progress?.currentPdf && progress.currentPage > 0 && (
              <p>
                Current: {progress.currentPdf} | page {progress.currentPage}/
                {progress.totalPagesForCurrent}
              </p>
            )}
          </div>
        </section>

        <section className="panel">
          <h2>3) Download Results</h2>
          <p className="muted">
            Output columns: <code>pdf_name</code>, <code>paragraph</code>,
            <code> paragraph_index</code>, <code>page_number</code>,
            <code> section_heading</code>, <code>notes</code>,{" "}
            <code>confidence</code>.
          </p>

          <div className="inline-actions">
            <button
              type="button"
              onClick={() => downloadCsv(rows, buildDownloadName("csv"))}
              disabled={rows.length === 0}
            >
              Download CSV
            </button>
            <button
              type="button"
              onClick={() => downloadXlsx(rows, buildDownloadName("xlsx"))}
              disabled={rows.length === 0}
            >
              Download XLSX
            </button>
          </div>

          <p className="muted">Rows ready: {rows.length}</p>
        </section>

        <section className="panel full-width">
          <h2>4) Extraction Quality Knobs</h2>
          <p className="muted">
            Tune minimum paragraph quality to reduce short fragments and headings.
          </p>

          <div className="split">
            <label className="field">
              <span>Minimum words per paragraph</span>
              <input
                type="number"
                min={1}
                max={100}
                step={1}
                value={qualitySettings.minWordsPerParagraph}
                onChange={(event) =>
                  setQualitySettings((previous) =>
                    sanitizeQualitySettings({
                      ...previous,
                      minWordsPerParagraph: toNumber(
                        event.target.value,
                        previous.minWordsPerParagraph,
                      ),
                    }),
                  )
                }
                disabled={isRunning}
              />
            </label>
            <label className="field">
              <span>Minimum alphabetic characters</span>
              <input
                type="number"
                min={1}
                max={600}
                step={1}
                value={qualitySettings.minAlphaCharsPerParagraph}
                onChange={(event) =>
                  setQualitySettings((previous) =>
                    sanitizeQualitySettings({
                      ...previous,
                      minAlphaCharsPerParagraph: toNumber(
                        event.target.value,
                        previous.minAlphaCharsPerParagraph,
                      ),
                    }),
                  )
                }
                disabled={isRunning}
              />
            </label>
            <label className="field">
              <span>Short paragraph word threshold</span>
              <input
                type="number"
                min={1}
                max={200}
                step={1}
                value={qualitySettings.shortParagraphWordThreshold}
                onChange={(event) =>
                  setQualitySettings((previous) =>
                    sanitizeQualitySettings({
                      ...previous,
                      shortParagraphWordThreshold: toNumber(
                        event.target.value,
                        previous.shortParagraphWordThreshold,
                      ),
                    }),
                  )
                }
                disabled={isRunning}
              />
            </label>
          </div>

          <label className="checkbox">
            <input
              type="checkbox"
              checked={qualitySettings.requireSentenceTerminatorForShortParagraphs}
              onChange={(event) =>
                setQualitySettings((previous) => ({
                  ...previous,
                  requireSentenceTerminatorForShortParagraphs:
                    event.target.checked,
                }))
              }
              disabled={isRunning}
            />
            <span>Require punctuation on short paragraphs</span>
          </label>

          <div className="inline-actions">
            <button
              type="button"
              onClick={() => setQualitySettings(DEFAULT_QUALITY_SETTINGS)}
              disabled={isRunning}
            >
              Reset Quality Defaults
            </button>
          </div>
        </section>

        <section className="panel full-width">
          <h2>5) Prompt Templates (Editable)</h2>
          <p className="muted">
            These prompts control inclusion/exclusion logic. Tune them to favor
            full sentence paragraphs and suppress short heading fragments.
          </p>

          <div className="inline-actions">
            <button
              type="button"
              onClick={() => promptFileInputRef.current?.click()}
              disabled={isRunning}
            >
              Import .md
            </button>
            <button
              type="button"
              onClick={handlePromptDownload}
              disabled={isRunning}
            >
              Export .md
            </button>
            <button
              type="button"
              onClick={() => {
                setPromptConfig(DEFAULT_PROMPT_CONFIG);
                appendLog("info", "Reset prompts to defaults.");
              }}
              disabled={isRunning}
            >
              Reset Defaults
            </button>
            <input
              ref={promptFileInputRef}
              type="file"
              accept=".md,text/markdown,text/plain"
              hidden
              onChange={(event) => {
                void handlePromptFileImport(event.target.files);
                event.currentTarget.value = "";
              }}
            />
          </div>

          <label className="field">
            <span>Text-layer filter prompt</span>
            <textarea
              value={promptConfig.textFilterSystem}
              onChange={(event) =>
                setPromptConfig((previous) => ({
                  ...previous,
                  textFilterSystem: event.target.value,
                }))
              }
              rows={10}
              disabled={isRunning}
            />
          </label>

          <label className="field">
            <span>Vision OCR page prompt</span>
            <textarea
              value={promptConfig.visionSystem}
              onChange={(event) =>
                setPromptConfig((previous) => ({
                  ...previous,
                  visionSystem: event.target.value,
                }))
              }
              rows={10}
              disabled={isRunning}
            />
          </label>
        </section>

        <section className="panel full-width">
          <h2>Status Log</h2>
          <div className="log-panel">
            {logs.length === 0 && <p className="muted">No status messages yet.</p>}
            {logs.map((entry) => (
              <div key={entry.id} className={`log-entry ${entry.level}`}>
                <span>[{entry.timestamp}]</span>
                <span>{entry.level.toUpperCase()}</span>
                <span>{entry.message}</span>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
