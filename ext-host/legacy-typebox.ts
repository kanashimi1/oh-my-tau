// Vendored copy of pi-coding-agent's `src/extensibility/legacy-typebox.ts` (keep
// behavior in sync with omp) with TWO changes, both import-cost only:
// - the original imports `@oh-my-pi/pi-ai/utils/schema`, whose barrel re-exports
//   ~14 schema modules and costs ~40 MB of RSS in the extension host. We only
//   need two leaf functions (`upgradeJsonSchemaTo202012`,
//   `validateJsonSchemaValue`), imported here directly from their source files.
// - `json-schema-validator.ts` alone is ~18 MB RSS / ~55 ms to compile and only
//   runs when a value is validated against a `Type.Unsafe` / raw-JSON-schema
//   property, which most sessions never do: its path is resolved eagerly (a
//   missing file still fails at `api.typebox` access, as before) but the module
//   is required on the first validation.
// Everything below the imports is verbatim omp code.
//
// `createRequire` anchored at the omp package keeps node_modules resolution
// identical to the original (omptype, pi-utils resolve from omp's install tree).
import { homedir } from "node:os";
import { join } from "node:path";
import { createRequire } from "node:module";

const PKG_ROOT =
  process.env.OMP_PKG_ROOT ?? join(homedir(), ".bun/install/global/node_modules/@oh-my-pi/pi-coding-agent");
const piRequire = createRequire(join(PKG_ROOT, "src", "extensibility", "legacy-typebox.ts"));

const { type } = piRequire("@oh-my-pi/omptype");
const { IR_BRAND } = piRequire("@oh-my-pi/omptype/ir");
const { Type: OmpType } = piRequire("@oh-my-pi/omptype/typebox");
const { upgradeJsonSchemaTo202012 } = piRequire(join(PKG_ROOT, "..", "pi-ai", "src", "utils", "schema", "draft.ts"));
// Resolved now (fails fast like the eager import did), loaded on first use.
const VALIDATOR_PATH = piRequire.resolve(join(PKG_ROOT, "..", "pi-ai", "src", "utils", "schema", "json-schema-validator.ts"));
type ValidateFn = (
  schema: unknown,
  data: unknown,
) => { success: boolean; issues: { path: (string | number)[]; message: string }[] };
let validatorFn: ValidateFn | undefined;
const validateJsonSchemaValue: ValidateFn = (schema, data) =>
  (validatorFn ??= piRequire(VALIDATOR_PATH).validateJsonSchemaValue as ValidateFn)(schema, data);

type AnySchema = unknown;
type ObjectOpts = { additionalProperties?: unknown; [k: string]: unknown };
type OmpTypeBuilder = typeof OmpType;
type TUnsafe<T> = AnySchema & { [IR_BRAND]?: unknown } & T;

const VALIDATION_FAILURE = Symbol("omp.typebox.validationFailure");

interface ValidationFailure {
  message: string;
  readonly [VALIDATION_FAILURE]: true;
}

interface SafeParseSuccess<T> {
  success: true;
  data: T;
}

interface SafeParseFailure {
  success: false;
  error: ValidationFailure;
}

type LegacyUnsafeSchema<T> = TUnsafe<T> & {
  __validator(data: unknown): T | ValidationFailure;
  safeParse(input: unknown): SafeParseSuccess<T> | SafeParseFailure;
};

function isValidationFailure<T>(result: T | ValidationFailure): result is ValidationFailure {
  return typeof result === "object" && result !== null && VALIDATION_FAILURE in result;
}

function isRuntimeSchema(value: unknown): value is AnySchema {
  return typeof value === "function";
}

function lowerEmbeddedSchemas(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(lowerEmbeddedSchemas);
  if (typeof node !== "object" || node === null) return node;
  const source = node as Record<string, unknown>;
  if (IR_BRAND in source && typeof (source as { toJsonSchema?: unknown }).toJsonSchema === "function") {
    return (source as { toJsonSchema: () => unknown }).toJsonSchema();
  }
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(source)) out[key] = lowerEmbeddedSchemas(value);
  return out;
}

function defineHidden(target: object, key: PropertyKey, value: unknown): void {
  Object.defineProperty(target, key, {
    value,
    writable: true,
    configurable: true,
  });
}

function unsafe<T = unknown>(jsonSchema: Record<string, unknown> = {}): LegacyUnsafeSchema<T> {
  const document = lowerEmbeddedSchemas(jsonSchema) as Record<string, unknown>;
  const upgradedSchema = upgradeJsonSchemaTo202012(structuredClone(document));
  const validate = (data: unknown): T | ValidationFailure => {
    const result = validateJsonSchemaValue(upgradedSchema, data);
    if (result.success) return data as T;
    let message = "";
    for (const issue of result.issues) {
      if (message) message += "; ";
      message += issue.path.length > 0 ? `${issue.path.join(".")}: ${issue.message}` : issue.message;
    }
    const failure = { message: message || "Invalid value" } as ValidationFailure;
    defineHidden(failure, VALIDATION_FAILURE, true);
    return failure;
  };
  const runtime = (type as { unknown: { narrow: (fn: (data: unknown, ctx: { mustBe: (m: string) => false }) => boolean) => unknown } }).unknown.narrow((data, ctx) => {
    const result = validate(data);
    return isValidationFailure(result) ? ctx.mustBe(result.message) : true;
  });
  const schema = type.withJsonSchema(runtime, document) as unknown as LegacyUnsafeSchema<T>;
  defineHidden(schema, "__validator", validate);
  defineHidden(schema, "safeParse", (input: unknown): SafeParseSuccess<T> | SafeParseFailure => {
    const result = validate(input);
    return isValidationFailure(result) ? { success: false, error: result } : { success: true, data: result };
  });
  return schema;
}

const object = ((properties: Record<string, unknown>, opts?: ObjectOpts) => {
  let normalizedOpts = opts;
  const additionalProperties: unknown = opts?.additionalProperties;
  if (
    additionalProperties !== undefined &&
    typeof additionalProperties !== "boolean" &&
    !isRuntimeSchema(additionalProperties)
  ) {
    normalizedOpts = {
      ...opts,
      additionalProperties: unsafe(additionalProperties as Record<string, unknown>),
    };
  }

  let hasRawProperty = false;
  for (const key in properties) {
    if (!isRuntimeSchema(properties[key])) {
      hasRawProperty = true;
      break;
    }
  }

  if (!hasRawProperty) {
    return (OmpType as { Object: (p: Record<string, AnySchema>, o?: ObjectOpts) => unknown }).Object(
      properties as Record<string, AnySchema>,
      normalizedOpts,
    );
  }

  const normalizedProperties: Record<string, AnySchema> = {};
  for (const key in properties) {
    const property = properties[key];
    normalizedProperties[key] = isRuntimeSchema(property) ? property : unsafe(property as Record<string, unknown>);
  }
  if (additionalProperties !== undefined && typeof additionalProperties !== "boolean") {
    const { additionalProperties: _, ...objectOpts } = opts ?? {};
    const document = (OmpType as { Object: (p: Record<string, AnySchema>, o?: ObjectOpts) => { toJsonSchema: () => Record<string, unknown> } }).Object(
      normalizedProperties,
      objectOpts,
    ).toJsonSchema();
    document.additionalProperties = isRuntimeSchema(additionalProperties)
      ? (additionalProperties as { toJsonSchema: () => unknown }).toJsonSchema()
      : lowerEmbeddedSchemas(additionalProperties);
    return unsafe(document);
  }
  return (OmpType as { Object: (p: Record<string, AnySchema>, o?: ObjectOpts) => unknown }).Object(
    normalizedProperties,
    normalizedOpts,
  );
}) as unknown as typeof OmpType.Object;

export const Type = { ...OmpType, Object: object, Unsafe: unsafe } as unknown as OmpTypeBuilder;
export type TypeBuilder = OmpTypeBuilder;

const legacyTypeBox: { Type: OmpTypeBuilder } = { Type };
export default legacyTypeBox;
