import type { BackendConfig, OllamaGenerationSettings } from "../types";

export type MessageContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

export type ChatMessage = {
  role: "system" | "user";
  content: string | MessageContentPart[];
};

export interface NativeToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface ChatCompletionRequestOptions {
  responseFormat?: "json_object";
  tool?: NativeToolDefinition;
}

export class HttpError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "HttpError";
    this.status = status;
  }
}

const jsonHeaders = {
  "Content-Type": "application/json",
};

function dataUrlToBase64(url: string): string | null {
  const match = url.match(/^data:[^;]+;base64,(.+)$/);
  if (!match?.[1]) return null;
  return match[1];
}

function toOllamaMessages(messages: ChatMessage[]): Array<{
  role: "system" | "user";
  content: string;
  images?: string[];
}> {
  return messages.map((message) => {
    if (typeof message.content === "string") {
      return {
        role: message.role,
        content: message.content,
      };
    }

    const textParts: string[] = [];
    const images: string[] = [];

    for (const part of message.content) {
      if (part.type === "text") {
        if (part.text.trim()) textParts.push(part.text);
        continue;
      }

      if (part.type === "image_url") {
        const base64 = dataUrlToBase64(part.image_url.url);
        if (base64) images.push(base64);
      }
    }

    const payload: {
      role: "system" | "user";
      content: string;
      images?: string[];
    } = {
      role: message.role,
      content:
        textParts.join("\n").trim() || "Analyze the provided page image.",
    };

    if (images.length > 0) {
      payload.images = images;
    }

    return payload;
  });
}

function sanitizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function toDevProxyBaseUrlIfNeeded(baseUrl: string): string {
  const sanitized = sanitizeBaseUrl(baseUrl);
  if (typeof window === "undefined") {
    return sanitized;
  }

  const host = window.location.hostname;
  const isLocalDevHost = host === "localhost" || host === "127.0.0.1";
  if (!isLocalDevHost) {
    return sanitized;
  }

  try {
    const parsed = new URL(sanitized);
    if (parsed.protocol === "https:" && parsed.hostname === "api.openai.com") {
      const path = parsed.pathname.replace(/\/+$/, "");
      if (path === "/v1") {
        return "/__proxy_openai/v1";
      }
      if (path === "" || path === "/") {
        return "/__proxy_openai";
      }
    }
  } catch {
    // Ignore invalid URL and use user-provided value as-is.
  }

  return sanitized;
}

function openAiPathCandidates(pathWithoutVersion: string): string[] {
  return [pathWithoutVersion, `/v1${pathWithoutVersion}`];
}

async function fetchJsonWithPathFallback(
  baseUrl: string,
  paths: string[],
  init: RequestInit,
): Promise<unknown> {
  const resolvedBaseUrl = toDevProxyBaseUrlIfNeeded(baseUrl);
  let lastError: Error | null = null;

  for (const path of paths) {
    const url = `${sanitizeBaseUrl(resolvedBaseUrl)}${path}`;
    try {
      const response = await fetch(url, init);

      if (response.ok) {
        return await response.json();
      }

      if (response.status === 404) {
        continue;
      }

      const bodyText = await response.text();
      throw new HttpError(
        response.status,
        `Request failed (${response.status}) at ${path}: ${bodyText.slice(0, 240)}`,
      );
    } catch (error) {
      if (error instanceof HttpError && error.status === 404) {
        continue;
      }
      if (error instanceof TypeError) {
        lastError = new Error(
          `Network/CORS failure while calling ${url}. If you are using OpenAI from the browser, run via local dev server proxy (npm run dev) or use a backend relay.`,
        );
        continue;
      }
      lastError =
        error instanceof Error ? error : new Error("Unknown network error");
    }
  }

  throw (
    lastError ??
    new Error(
      `Endpoint did not respond with a supported route. Tried: ${paths.join(", ")}`,
    )
  );
}

export async function listModels(
  config: Omit<BackendConfig, "model">,
  signal?: AbortSignal,
): Promise<string[]> {
  if (!config.baseUrl.trim()) {
    throw new Error("Base URL is required.");
  }

  if (config.kind === "openai") {
    const headers: Record<string, string> = { ...jsonHeaders };
    if (config.apiKey.trim()) {
      headers.Authorization = `Bearer ${config.apiKey.trim()}`;
    }

    const data = await fetchJsonWithPathFallback(
      config.baseUrl,
      openAiPathCandidates("/models"),
      {
        method: "GET",
        headers,
        signal,
      },
    );

    const ids = Array.isArray((data as { data?: unknown[] }).data)
      ? (data as { data: Array<{ id?: string }> }).data
          .map((item) => item.id)
          .filter((id): id is string => Boolean(id))
      : [];

    if (ids.length === 0) {
      throw new Error(
        "No models were returned by the endpoint. Enter a model ID manually.",
      );
    }

    return ids.sort();
  }

  const data = await fetchJsonWithPathFallback(config.baseUrl, ["/api/tags"], {
    method: "GET",
    headers: jsonHeaders,
    signal,
  });

  const names = Array.isArray((data as { models?: unknown[] }).models)
    ? (data as { models: Array<{ name?: string }> }).models
        .map((item) => item.name)
        .filter((name): name is string => Boolean(name))
    : [];

  if (names.length === 0) {
    throw new Error(
      "No Ollama models were returned by /api/tags. Enter a model name manually.",
    );
  }

  return names.sort();
}

function parseOpenAiContent(data: unknown): string {
  const choice = (data as { choices?: Array<{ message?: { content?: unknown } }> })
    .choices?.[0];
  const content = choice?.message?.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    const text = content
      .map((item) => (item as { text?: unknown }).text)
      .filter((value): value is string => typeof value === "string")
      .join("");
    if (text) return text;
  }
  throw new Error("OpenAI-compatible endpoint returned an unexpected response shape.");
}

function parseOllamaContent(data: unknown): string {
  const message = (data as {
    message?: {
      content?: unknown;
      tool_calls?: Array<{
        function?: { arguments?: unknown };
      }>;
    };
  }).message;

  const firstToolCallArgs = message?.tool_calls?.[0]?.function?.arguments;
  if (typeof firstToolCallArgs === "string") {
    return firstToolCallArgs;
  }
  if (
    firstToolCallArgs &&
    typeof firstToolCallArgs === "object" &&
    !Array.isArray(firstToolCallArgs)
  ) {
    return JSON.stringify(firstToolCallArgs);
  }

  const content = message?.content;
  if (typeof content === "string") {
    return content;
  }
  const fallback = (data as { response?: unknown }).response;
  if (typeof fallback === "string") {
    return fallback;
  }
  throw new Error("Ollama endpoint returned an unexpected response shape.");
}

function buildOllamaOptions(
  ollama: OllamaGenerationSettings | undefined,
): Record<string, number> {
  return {
    temperature: ollama?.temperature ?? 0,
    top_p: ollama?.topP ?? 0.9,
    top_k: ollama?.topK ?? 40,
    min_p: ollama?.minP ?? 0,
    repeat_penalty: ollama?.repeatPenalty ?? 1.1,
    num_ctx: ollama?.contextSize ?? 8192,
  };
}

function shouldRetryWithoutTools(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const message = error.message.toLowerCase();
  return (
    message.includes("tool") ||
    message.includes("unsupported") ||
    message.includes("unknown field")
  );
}

export async function runChatCompletion(
  config: BackendConfig,
  messages: ChatMessage[],
  signal?: AbortSignal,
  requestOptions?: ChatCompletionRequestOptions,
): Promise<string> {
  if (!config.model.trim()) {
    throw new Error("Model is required.");
  }

  if (config.kind === "openai") {
    const headers: Record<string, string> = { ...jsonHeaders };
    if (config.apiKey.trim()) {
      headers.Authorization = `Bearer ${config.apiKey.trim()}`;
    }

    const openAiBody: {
      model: string;
      messages: ChatMessage[];
      response_format?: { type: "json_object" };
    } = {
      model: config.model,
      messages,
    };

    if (requestOptions?.responseFormat === "json_object") {
      openAiBody.response_format = { type: "json_object" };
    }

    const data = await fetchJsonWithPathFallback(
      config.baseUrl,
      openAiPathCandidates("/chat/completions"),
      {
        method: "POST",
        headers,
        signal,
        body: JSON.stringify(openAiBody),
      },
    );
    return parseOpenAiContent(data);
  }

  const body: {
    model: string;
    stream: boolean;
    messages: Array<{ role: "system" | "user"; content: string; images?: string[] }>;
    options: Record<string, number>;
    format?: "json";
    tools?: Array<{
      type: "function";
      function: NativeToolDefinition;
    }>;
  } = {
    model: config.model,
    stream: false,
    messages: toOllamaMessages(messages),
    options: buildOllamaOptions(config.ollama),
  };

  if (requestOptions?.responseFormat === "json_object") {
    body.format = "json";
  }

  if (config.ollama?.useNativeToolCalling && requestOptions?.tool) {
    body.tools = [
      {
        type: "function",
        function: requestOptions.tool,
      },
    ];
  }

  let data: unknown;
  try {
    data = await fetchJsonWithPathFallback(config.baseUrl, ["/api/chat"], {
      method: "POST",
      headers: jsonHeaders,
      signal,
      body: JSON.stringify(body),
    });
  } catch (error) {
    if (!body.tools || !shouldRetryWithoutTools(error)) {
      throw error;
    }

    const fallbackBody = { ...body };
    delete fallbackBody.tools;

    data = await fetchJsonWithPathFallback(config.baseUrl, ["/api/chat"], {
      method: "POST",
      headers: jsonHeaders,
      signal,
      body: JSON.stringify(fallbackBody),
    });
  }

  return parseOllamaContent(data);
}
