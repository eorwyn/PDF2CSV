export async function mapWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  worker: (item: T, index: number) => Promise<R>,
  signal?: AbortSignal,
): Promise<R[]> {
  if (items.length === 0) {
    return [];
  }

  const boundedConcurrency = Math.max(1, Math.floor(concurrency));
  const results = new Array<R>(items.length);
  let currentIndex = 0;

  async function runWorker(): Promise<void> {
    while (true) {
      if (signal?.aborted) {
        throw new DOMException("Aborted", "AbortError");
      }

      const index = currentIndex;
      currentIndex += 1;
      if (index >= items.length) {
        return;
      }
      results[index] = await worker(items[index], index);
    }
  }

  const workers = Array.from(
    { length: Math.min(boundedConcurrency, items.length) },
    () => runWorker(),
  );

  await Promise.all(workers);
  return results;
}
