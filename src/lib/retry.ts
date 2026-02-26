import { HttpError } from "./backend";

export interface RetryOptions {
  retries: number;
  baseDelayMs?: number;
  signal?: AbortSignal;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, ms);
    if (!signal) return;
    signal.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

function isTransientError(error: unknown): boolean {
  if (error instanceof DOMException && error.name === "AbortError") {
    return false;
  }

  if (error instanceof HttpError) {
    return error.status === 429 || error.status >= 500;
  }

  return true;
}

export async function withRetries<T>(
  operation: () => Promise<T>,
  options: RetryOptions,
): Promise<T> {
  const retries = Math.max(0, options.retries);
  const baseDelayMs = options.baseDelayMs ?? 800;

  let attempt = 0;
  let lastError: unknown;

  while (attempt <= retries) {
    if (options.signal?.aborted) {
      throw new DOMException("Aborted", "AbortError");
    }

    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (attempt >= retries || !isTransientError(error)) {
        break;
      }
      const jitter = Math.floor(Math.random() * 220);
      await sleep(baseDelayMs * (attempt + 1) + jitter, options.signal);
      attempt += 1;
    }
  }

  throw (lastError instanceof Error
    ? lastError
    : new Error("Operation failed after retries"));
}
