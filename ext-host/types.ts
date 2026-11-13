// types.ts — shared wire/ext/delta types (mechanical split from host.ts).
export type Json = Record<string, unknown>;
export type Handler = (event: Json, ctx: unknown) => unknown;
export interface Ext {
  path: string;
  name: string;
  handlers: Map<string, Handler[]>;
  tools: Map<string, Json>;
  commands: Map<string, Json>;
  flags: Map<string, Json>;
  label?: string;
}
export const TIMEOUT = Symbol("timeout");
export interface HandlerRecord { ext: string; event: string; tool?: string; ms: number; action: string; error?: string }
export type UpdateChain = { id: number; seq: number; message: Json };
export class DeltaError extends Error {}
export const STREAM_KEY: Record<string, string> = { text: "text", thinking: "thinking", toolCall: "arguments" };
