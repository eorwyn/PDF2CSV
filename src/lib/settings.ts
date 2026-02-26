import type {
  ExtractionQualitySettings,
  OllamaGenerationSettings,
} from "../types";

export const DEFAULT_QUALITY_SETTINGS: ExtractionQualitySettings = {
  minWordsPerParagraph: 6,
  minAlphaCharsPerParagraph: 18,
  shortParagraphWordThreshold: 12,
  requireSentenceTerminatorForShortParagraphs: true,
};

export const DEFAULT_OLLAMA_SETTINGS: OllamaGenerationSettings = {
  temperature: 0,
  topP: 0.9,
  topK: 40,
  minP: 0,
  repeatPenalty: 1.1,
  contextSize: 8192,
  useNativeToolCalling: false,
};

function toFiniteNumber(value: unknown, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  return value;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function sanitizeQualitySettings(
  input: Partial<ExtractionQualitySettings> | undefined,
): ExtractionQualitySettings {
  return {
    minWordsPerParagraph: Math.floor(
      clamp(
        toFiniteNumber(
          input?.minWordsPerParagraph,
          DEFAULT_QUALITY_SETTINGS.minWordsPerParagraph,
        ),
        1,
        100,
      ),
    ),
    minAlphaCharsPerParagraph: Math.floor(
      clamp(
        toFiniteNumber(
          input?.minAlphaCharsPerParagraph,
          DEFAULT_QUALITY_SETTINGS.minAlphaCharsPerParagraph,
        ),
        1,
        600,
      ),
    ),
    shortParagraphWordThreshold: Math.floor(
      clamp(
        toFiniteNumber(
          input?.shortParagraphWordThreshold,
          DEFAULT_QUALITY_SETTINGS.shortParagraphWordThreshold,
        ),
        1,
        200,
      ),
    ),
    requireSentenceTerminatorForShortParagraphs:
      input?.requireSentenceTerminatorForShortParagraphs ??
      DEFAULT_QUALITY_SETTINGS.requireSentenceTerminatorForShortParagraphs,
  };
}

export function sanitizeOllamaSettings(
  input: Partial<OllamaGenerationSettings> | undefined,
): OllamaGenerationSettings {
  return {
    temperature: clamp(
      toFiniteNumber(input?.temperature, DEFAULT_OLLAMA_SETTINGS.temperature),
      0,
      2,
    ),
    topP: clamp(toFiniteNumber(input?.topP, DEFAULT_OLLAMA_SETTINGS.topP), 0, 1),
    topK: Math.floor(
      clamp(toFiniteNumber(input?.topK, DEFAULT_OLLAMA_SETTINGS.topK), 0, 500),
    ),
    minP: clamp(toFiniteNumber(input?.minP, DEFAULT_OLLAMA_SETTINGS.minP), 0, 1),
    repeatPenalty: clamp(
      toFiniteNumber(
        input?.repeatPenalty,
        DEFAULT_OLLAMA_SETTINGS.repeatPenalty,
      ),
      0.5,
      3,
    ),
    contextSize: Math.floor(
      clamp(
        toFiniteNumber(input?.contextSize, DEFAULT_OLLAMA_SETTINGS.contextSize),
        256,
        262144,
      ),
    ),
    useNativeToolCalling:
      input?.useNativeToolCalling ?? DEFAULT_OLLAMA_SETTINGS.useNativeToolCalling,
  };
}
