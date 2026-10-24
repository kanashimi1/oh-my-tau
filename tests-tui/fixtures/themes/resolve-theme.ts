import { resolveVarRefs } from './resolve-vars';

interface ColorValue {
  type: 'hex' | 'ansi' | 'var_ref';
  value: string | number;
  isBg?: boolean;
}

interface ResolvedColor {
  hex: string;
  ansi?: string;
}

function parseColorValue(value: string | number): ColorValue {
  if (typeof value === 'number') {
    return { type: 'ansi', value, isBg: value >= 16 && value <= 255 };
  }
  
  if (value.startsWith('#')) {
    return { type: 'hex', value, isBg: false };
  }
  
  if (value.startsWith('$')) {
    return { type: 'var_ref', value: value.substring(1), isBg: false };
  }
  
  if (value === '') {
    return { type: 'hex', value: '', isBg: false };
  }
  
  if (value.includes('okhsl(')) {
    return { type: 'hex', value, isBg: false };
  }
  
  // Assume it's a variable reference
  return { type: 'var_ref', value, isBg: false };
}

export function resolveThemeColors<T extends Record<string, string | number>>(
  colors: T,
  vars: Record<string, string | number> = {}
): Record<keyof T, string | number> {
  const resolved: Record<string, string | number> = {};
  
  for (const [key, value] of Object.entries(colors)) {
    const parsed = parseColorValue(value);
    let resolvedValue = value;
    
    if (parsed.type === 'hex') {
      resolvedValue = parsed.value;
    } else if (parsed.type === 'var_ref') {
      // Resolve variable reference
      const varValue = vars[parsed.value];
      if (varValue !== undefined) {
        resolvedValue = resolveVarRefs(varValue, vars);
      } else {
        console.warn(`Variable reference '${parsed.value}' not found for color '${key}'`);
        resolvedValue = value;
      }
    }
    
    resolved[key] = resolvedValue;
  }
  
  return resolved as Record<keyof T, string | number>;
}

export function resolveVarRefs(value: string | number, vars: Record<string, string | number>): string | number {
  const parsed = parseColorValue(value);
  
  if (parsed.type === 'hex' || parsed.type === 'ansi') {
    return value;
  } else if (parsed.type === 'var_ref') {
    const varValue = vars[parsed.value];
    if (varValue !== undefined) {
      return resolveVarRefs(varValue, vars);
    } else {
      console.warn(`Variable reference '${parsed.value}' not found`);
      return value;
    }
  }
  
  return value;
}