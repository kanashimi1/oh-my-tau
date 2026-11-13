/**
 * SessionState: explicit lifecycle for the extension host.
 *
 * Replaces the scattered `nowaitTail`, `loading`, `nowaitBlocked` module
 * globals with a single state machine. Valid transitions:
 *   Idle → Loading → Ready → ShuttingDown
 *   Ready → Ready (normal operation)
 *   ShuttingDown → terminal (no transitions out)
 */
export type SessionPhase = "idle" | "loading" | "ready" | "shutting_down";

export class SessionState {
  private phase: SessionPhase = "idle";
  private loadingPromise: Promise<void> = Promise.resolve();
  private nowaitTail: Promise<void> = Promise.resolve();
  private nowaitBlocked = 0;
  private nowaitWakers = new Set<() => void>();

  get current(): SessionPhase {
    return this.phase;
  }

  isReady(): boolean {
    return this.phase === "ready";
  }

  isShuttingDown(): boolean {
    return this.phase === "shutting_down";
  }

  enterLoading(): Promise<void> {
    this.phase = "loading";
    return this.loadingPromise;
  }

  setReady(): void {
    this.phase = "ready";
  }

  enterShuttingDown(): void {
    this.phase = "shutting_down";
  }

  chainNowait(run: () => Promise<unknown>): Promise<void> {
    const prev = this.nowaitTail;
    const chained = (async () => {
      await prev;
      return run();
    })();
    this.nowaitTail = chained.then(
      () => undefined,
      () => undefined,
    );
    return this.nowaitTail;
  }

  awaitNowait(): Promise<void> {
    return this.nowaitTail;
  }

  addWaker(wake: () => void): void {
    this.nowaitWakers.add(wake);
  }

  removeWaker(wake: () => void): void {
    this.nowaitWakers.delete(wake);
  }

  wakeAll(): void {
    for (const wake of this.nowaitWakers) wake();
    this.nowaitWakers.clear();
  }

  blockedCount(): number {
    return this.nowaitBlocked;
  }

  incBlocked(): void {
    this.nowaitBlocked++;
  }

  decBlocked(): void {
    this.nowaitBlocked--;
  }

  awaitLoading(): Promise<void> {
    return this.loadingPromise;
  }

  setLoading(p: Promise<void>): void {
    this.loadingPromise = p.catch(() => undefined);
  }
}
