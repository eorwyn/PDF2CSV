import { mapWithConcurrency } from "./concurrency";
import { runChatCompletion, type NativeToolDefinition } from "./backend";
import { extractPdfCandidates, iteratePdfPageImages } from "./pdf";
import { DEFAULT_PROMPT_CONFIG } from "./prompts";
import { withRetries } from "./retry";
import { sanitizeQualitySettings } from "./settings";
import type {
  BackendConfig,
  ChunkDecision,
  ExtractionQualitySettings,
  ExtractionRow,
  KeepDecision,
  ParagraphCandidate,
  PromptConfig,
  RunProgress,
} from "../types";

interface ExtractionCallbacks {
  onLog?: (level: "info" | "warning" | "error", message: string) => void;
  onProgress?: (progress: RunProgress) => void;
}

interface VisionParagraphDecision {
  text: string;
  section_heading?: string;
  note?: string;
  confidence?: number;
  possible_boilerplate?: boolean;
}

interface VisionPageDecision {
  paragraphs: VisionParagraphDecision[];
  warnings: string[];
}

export interface RunExtractionOptions extends ExtractionCallbacks {
  config: BackendConfig;
  prompts?: PromptConfig;
  quality?: ExtractionQualitySettings;
  fileConcurrency: number;
  retries: number;
  signal?: AbortSignal;
}

const TEXT_FILTER_TOOL: NativeToolDefinition = {
  name: "return_text_filter_decision",
  description:
    "Return paragraph IDs to keep as main narrative content for qualitative coding.",
  parameters: {
    type: "object",
    properties: {
      keep: {
        type: "array",
        items: {
          type: "object",
          properties: {
            id: { type: "string" },
            possible_boilerplate: { type: "boolean" },
            section_heading: { type: "string" },
            note: { type: "string" },
            confidence: { type: "number" },
          },
          required: ["id"],
        },
      },
      warnings: {
        type: "array",
        items: { type: "string" },
      },
    },
    required: ["keep"],
  },
};

const VISION_PAGE_TOOL: NativeToolDefinition = {
  name: "return_vision_page_paragraphs",
  description:
    "Return OCR paragraph extraction for one PDF page image, excluding non-core text.",
  parameters: {
    type: "object",
    properties: {
      paragraphs: {
        type: "array",
        items: {
          type: "object",
          properties: {
            text: { type: "string" },
            section_heading: { type: "string" },
            note: { type: "string" },
            possible_boilerplate: { type: "boolean" },
            confidence: { type: "number" },
          },
          required: ["text"],
        },
      },
      warnings: {
        type: "array",
        items: { type: "string" },
      },
    },
    required: ["paragraphs"],
  },
};

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }
}

function normalizeParagraph(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function isLikelyCodingParagraph(
  text: string,
  quality: ExtractionQualitySettings,
): boolean {
  const normalized = normalizeParagraph(text);
  if (!normalized) return false;

  const words = normalized.split(/\s+/).filter(Boolean);
  if (words.length < quality.minWordsPerParagraph) {
    return false;
  }

  const alphaChars = (normalized.match(/[A-Za-z]/g) ?? []).length;
  if (alphaChars < quality.minAlphaCharsPerParagraph) {
    return false;
  }

  if (!quality.requireSentenceTerminatorForShortParagraphs) {
    return true;
  }

  const hasSentenceEnding = /[.!?]["')\]]?$/.test(normalized);
  if (!hasSentenceEnding && words.length < quality.shortParagraphWordThreshold) {
    return false;
  }

  return true;
}

function chunkParagraphs(
  paragraphs: ParagraphCandidate[],
  maxChars = 7000,
): ParagraphCandidate[][] {
  const chunks: ParagraphCandidate[][] = [];
  let currentChunk: ParagraphCandidate[] = [];
  let currentLength = 0;

  for (const paragraph of paragraphs) {
    const serializedLength = paragraph.text.length + 120;
    if (
      currentChunk.length > 0 &&
      currentLength + serializedLength > maxChars
    ) {
      chunks.push(currentChunk);
      currentChunk = [];
      currentLength = 0;
    }
    currentChunk.push(paragraph);
    currentLength += serializedLength;
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk);
  }

  return chunks;
}

function extractJsonObject(raw: string): string {
  const fencedMatch = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fencedMatch?.[1]) {
    return fencedMatch[1].trim();
  }

  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start < 0 || end <= start) {
    throw new Error("Model output did not contain JSON.");
  }
  return raw.slice(start, end + 1);
}

function clampConfidence(value: unknown): number | undefined {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return undefined;
  }
  return Math.max(0, Math.min(1, value));
}

function normalizeDecision(raw: unknown): ChunkDecision {
  const parsed = raw as { keep?: unknown; warnings?: unknown };
  if (!Array.isArray(parsed.keep)) {
    throw new Error("Missing keep array in model JSON response.");
  }

  const keep: KeepDecision[] = parsed.keep
    .map((item) => {
      const record = item as Record<string, unknown>;
      if (typeof record.id !== "string" || !record.id.trim()) {
        return null;
      }
      return {
        id: record.id,
        section_heading:
          typeof record.section_heading === "string"
            ? record.section_heading.trim()
            : undefined,
        note:
          typeof record.note === "string" ? record.note.trim() : undefined,
        possible_boilerplate: Boolean(record.possible_boilerplate),
        confidence: clampConfidence(record.confidence),
      };
    })
    .filter((item): item is KeepDecision => item !== null);

  const warnings = Array.isArray(parsed.warnings)
    ? parsed.warnings
        .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
        .filter(Boolean)
    : [];

  return { keep, warnings };
}

function normalizeVisionDecision(raw: unknown): VisionPageDecision {
  const parsed = raw as { paragraphs?: unknown; warnings?: unknown };
  if (!Array.isArray(parsed.paragraphs)) {
    throw new Error("Missing paragraphs array in vision JSON response.");
  }

  const paragraphs: VisionParagraphDecision[] = parsed.paragraphs
    .map((item) => {
      const record = item as Record<string, unknown>;
      if (typeof record.text !== "string") {
        return null;
      }
      const text = record.text.trim();
      if (!text) {
        return null;
      }
      return {
        text,
        section_heading:
          typeof record.section_heading === "string"
            ? record.section_heading.trim()
            : undefined,
        note:
          typeof record.note === "string" ? record.note.trim() : undefined,
        possible_boilerplate: Boolean(record.possible_boilerplate),
        confidence: clampConfidence(record.confidence),
      };
    })
    .filter((item): item is VisionParagraphDecision => item !== null);

  const warnings = Array.isArray(parsed.warnings)
    ? parsed.warnings
        .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
        .filter(Boolean)
    : [];

  return { paragraphs, warnings };
}

async function extractPageParagraphsWithVision(
  config: BackendConfig,
  visionSystemPrompt: string,
  quality: ExtractionQualitySettings,
  pdfName: string,
  pageNumber: number,
  imageDataUrl: string,
  signal?: AbortSignal,
): Promise<VisionPageDecision> {
  const userContent = [
    {
      type: "text" as const,
      text: [
        `Document: ${pdfName}`,
        `Page: ${pageNumber}`,
        "Extract main-body paragraphs from this page image.",
        `Minimum words per paragraph: ${quality.minWordsPerParagraph}`,
        `Minimum alphabetic characters per paragraph: ${quality.minAlphaCharsPerParagraph}`,
        quality.requireSentenceTerminatorForShortParagraphs
          ? `If a paragraph has fewer than ${quality.shortParagraphWordThreshold} words, require terminal punctuation (.,!,?).`
          : "Terminal punctuation is optional for short paragraphs.",
      ].join("\n"),
    },
    {
      type: "image_url" as const,
      image_url: { url: imageDataUrl },
    },
  ];

  const responseText = await runChatCompletion(
    config,
    [
      { role: "system", content: visionSystemPrompt },
      { role: "user", content: userContent },
    ],
    signal,
    {
      responseFormat: "json_object",
      tool: VISION_PAGE_TOOL,
    },
  );

  const jsonText = extractJsonObject(responseText);
  const rawObject = JSON.parse(jsonText) as unknown;
  return normalizeVisionDecision(rawObject);
}

async function filterChunkWithLlm(
  config: BackendConfig,
  textFilterSystemPrompt: string,
  quality: ExtractionQualitySettings,
  chunk: ParagraphCandidate[],
  signal?: AbortSignal,
): Promise<ChunkDecision> {
  const input = chunk.map((paragraph) => ({
    id: paragraph.id,
    page_number: paragraph.pageNumber,
    text: paragraph.text,
  }));

  const userPrompt = [
    "Input paragraphs JSON:",
    JSON.stringify(input),
    `Minimum words per paragraph: ${quality.minWordsPerParagraph}`,
    `Minimum alphabetic characters per paragraph: ${quality.minAlphaCharsPerParagraph}`,
    quality.requireSentenceTerminatorForShortParagraphs
      ? `If paragraph is shorter than ${quality.shortParagraphWordThreshold} words, keep only if it ends with ., !, or ?.`
      : "Terminal punctuation is optional for short paragraphs.",
    "Return only JSON with IDs to keep.",
  ].join("\n");

  const responseText = await runChatCompletion(
    config,
    [
      { role: "system", content: textFilterSystemPrompt },
      { role: "user", content: userPrompt },
    ],
    signal,
    {
      responseFormat: "json_object",
      tool: TEXT_FILTER_TOOL,
    },
  );

  const jsonText = extractJsonObject(responseText);
  const rawObject = JSON.parse(jsonText) as unknown;
  return normalizeDecision(rawObject);
}

function dedupeRowsWithinPdf(rows: ExtractionRow[]): ExtractionRow[] {
  const seen = new Set<string>();
  const deduped: ExtractionRow[] = [];

  for (const row of rows) {
    const key = row.paragraph.replace(/\s+/g, " ").trim();
    if (!key) {
      deduped.push(row);
      continue;
    }
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(row);
  }

  return deduped.map((row, index) => ({ ...row, paragraph_index: index + 1 }));
}

function fallbackRowsFromChunk(
  pdfName: string,
  chunk: ParagraphCandidate[],
  errorMessage: string,
): ExtractionRow[] {
  return chunk.map((paragraph) => ({
    pdf_name: pdfName,
    paragraph: paragraph.text,
    paragraph_index: 0,
    page_number: paragraph.pageNumber,
    section_heading: "",
    notes: `LLM filtering failed for this chunk; included paragraph as fallback. Error: ${errorMessage}`,
    confidence: null,
  }));
}

async function processPdfWithVisionFallback(
  file: File,
  options: RunExtractionOptions,
  totalPdfs: number,
  getCompletedPdfs: () => number,
): Promise<ExtractionRow[]> {
  const visionSystemPrompt =
    options.prompts?.visionSystem?.trim() || DEFAULT_PROMPT_CONFIG.visionSystem;
  const qualitySettings = sanitizeQualitySettings(options.quality);

  const rows: ExtractionRow[] = [];
  let droppedByQuality = 0;

  await iteratePdfPageImages(
    file,
    async ({ pageNumber, totalPages, imageDataUrl }) => {
      throwIfAborted(options.signal);
      options.onProgress?.({
        totalPdfs,
        completedPdfs: getCompletedPdfs(),
        currentPdf: file.name,
        currentPage: pageNumber,
        totalPagesForCurrent: totalPages,
      });

      try {
        const decision = await withRetries(
          () =>
            extractPageParagraphsWithVision(
              options.config,
              visionSystemPrompt,
              qualitySettings,
              file.name,
              pageNumber,
              imageDataUrl,
              options.signal,
            ),
          { retries: options.retries, signal: options.signal },
        );

        decision.warnings.forEach((warning) =>
          options.onLog?.(
            "warning",
            `${file.name} page ${pageNumber}: ${warning}`,
          ),
        );

        if (decision.paragraphs.length === 0) {
          options.onLog?.(
            "warning",
            `${file.name} page ${pageNumber}: vision model returned no main-body paragraphs.`,
          );
          return;
        }

        decision.paragraphs.forEach((paragraph) => {
          if (!isLikelyCodingParagraph(paragraph.text, qualitySettings)) {
            droppedByQuality += 1;
            return;
          }

          const notes: string[] = [];
          if (paragraph.note) notes.push(paragraph.note);
          if (paragraph.possible_boilerplate) notes.push("possible boilerplate");

          rows.push({
            pdf_name: file.name,
            paragraph: paragraph.text,
            paragraph_index: 0,
            page_number: pageNumber,
            section_heading: paragraph.section_heading ?? "",
            notes: notes.join("; "),
            confidence: paragraph.confidence ?? null,
          });
        });
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          throw error;
        }
        const message =
          error instanceof Error ? error.message : "Unknown vision extraction error";
        options.onLog?.(
          "error",
          `${file.name} page ${pageNumber}: vision extraction failed after retries. ${message}`,
        );
        rows.push({
          pdf_name: file.name,
          paragraph: "",
          paragraph_index: 0,
          page_number: pageNumber,
          section_heading: "",
          notes: `Vision extraction failed on page ${pageNumber}: ${message}`,
          confidence: null,
        });
      }
    },
    options.signal,
  );

  if (rows.length === 0) {
    rows.push({
      pdf_name: file.name,
      paragraph: "",
      paragraph_index: 0,
      page_number: null,
      section_heading: "",
      notes:
        "No extractable paragraphs were found by text-layer parsing or vision OCR extraction.",
      confidence: null,
    });
  }

  if (droppedByQuality > 0) {
    options.onLog?.(
      "info",
      `${file.name}: dropped ${droppedByQuality} short/non-sentence fragments during quality filtering.`,
    );
  }

  const deduped = dedupeRowsWithinPdf(rows);
  const removedDuplicates = rows.length - deduped.length;
  if (removedDuplicates > 0) {
    options.onLog?.(
      "info",
      `${file.name}: removed ${removedDuplicates} exact duplicate paragraphs within this PDF.`,
    );
  }

  return deduped;
}

async function processSinglePdf(
  file: File,
  options: RunExtractionOptions,
  totalPdfs: number,
  getCompletedPdfs: () => number,
): Promise<ExtractionRow[]> {
  const textFilterSystemPrompt =
    options.prompts?.textFilterSystem?.trim() ||
    DEFAULT_PROMPT_CONFIG.textFilterSystem;
  const qualitySettings = sanitizeQualitySettings(options.quality);

  const parsed = await extractPdfCandidates(
    file,
    (page, totalPages) => {
      options.onProgress?.({
        totalPdfs,
        completedPdfs: getCompletedPdfs(),
        currentPdf: file.name,
        currentPage: page,
        totalPagesForCurrent: totalPages,
      });
    },
    options.signal,
  );

  parsed.warnings.forEach((warning) =>
    options.onLog?.("warning", `${file.name}: ${warning}`),
  );

  if (parsed.paragraphs.length === 0) {
    options.onLog?.(
      "warning",
      `${file.name}: no paragraph candidates remained after text-layer parsing.`,
    );
    options.onLog?.(
      "info",
      `${file.name}: attempting vision OCR fallback on rendered page images (requires a vision-capable model).`,
    );
    return processPdfWithVisionFallback(
      file,
      options,
      totalPdfs,
      getCompletedPdfs,
    );
  }

  const chunks = chunkParagraphs(parsed.paragraphs);
  const rows: ExtractionRow[] = [];
  let droppedByQuality = 0;

  for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex += 1) {
    throwIfAborted(options.signal);
    const chunk = chunks[chunkIndex];
    const chunkLabel = `chunk ${chunkIndex + 1}/${chunks.length}`;

    try {
      const decision = await withRetries(
        () =>
          filterChunkWithLlm(
            options.config,
            textFilterSystemPrompt,
            qualitySettings,
            chunk,
            options.signal,
          ),
        { retries: options.retries, signal: options.signal },
      );

      const candidateMap = new Map(chunk.map((item) => [item.id, item]));
      const used = new Set<string>();
      for (const keep of decision.keep) {
        const candidate = candidateMap.get(keep.id);
        if (!candidate || used.has(keep.id)) continue;
        if (!isLikelyCodingParagraph(candidate.text, qualitySettings)) {
          droppedByQuality += 1;
          continue;
        }
        used.add(keep.id);

        const notes: string[] = [];
        if (keep.note) notes.push(keep.note);
        if (keep.possible_boilerplate) notes.push("possible boilerplate");

        rows.push({
          pdf_name: parsed.pdfName,
          paragraph: candidate.text,
          paragraph_index: 0,
          page_number: candidate.pageNumber,
          section_heading: keep.section_heading ?? "",
          notes: notes.join("; "),
          confidence: keep.confidence ?? null,
        });
      }

      if ((decision.warnings?.length ?? 0) > 0) {
        options.onLog?.(
          "warning",
          `${file.name} ${chunkLabel}: ${decision.warnings?.join(" | ")}`,
        );
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw error;
      }
      const message =
        error instanceof Error ? error.message : "Unknown model error";
      options.onLog?.(
        "error",
        `${file.name} ${chunkLabel}: LLM filtering failed after retries. ${message}`,
      );
      rows.push(...fallbackRowsFromChunk(parsed.pdfName, chunk, message));
    }
  }

  const deduped = dedupeRowsWithinPdf(rows);
  const removedDuplicates = rows.length - deduped.length;
  if (removedDuplicates > 0) {
    options.onLog?.(
      "info",
      `${file.name}: removed ${removedDuplicates} exact duplicate paragraphs within this PDF.`,
    );
  }

  if (droppedByQuality > 0) {
    options.onLog?.(
      "info",
      `${file.name}: dropped ${droppedByQuality} short/non-sentence fragments during quality filtering.`,
    );
  }

  return deduped;
}

export async function runExtraction(
  files: File[],
  options: RunExtractionOptions,
): Promise<ExtractionRow[]> {
  const totalPdfs = files.length;
  let completedPdfs = 0;

  const perFileRows = await mapWithConcurrency(
    files,
    options.fileConcurrency,
    async (file) => {
      try {
        options.onLog?.("info", `Starting extraction for ${file.name}`);
        const rows = await processSinglePdf(file, options, totalPdfs, () => completedPdfs);
        options.onLog?.("info", `Finished ${file.name} with ${rows.length} paragraphs.`);
        return rows;
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          throw error;
        }

        const message =
          error instanceof Error ? error.message : "Unknown processing error";
        options.onLog?.("error", `${file.name}: failed to process file. ${message}`);
        return [
          {
            pdf_name: file.name,
            paragraph: "",
            paragraph_index: 1,
            page_number: null,
            section_heading: "",
            notes: `File processing failed: ${message}`,
            confidence: null,
          },
        ] as ExtractionRow[];
      } finally {
        completedPdfs += 1;
        options.onProgress?.({
          totalPdfs,
          completedPdfs,
          currentPdf: file.name,
          currentPage: 0,
          totalPagesForCurrent: 0,
        });
      }
    },
    options.signal,
  );

  return perFileRows.flat();
}
