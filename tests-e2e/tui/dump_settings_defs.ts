import { createSettingsHost } from "./src/config/settings-ui";
import { getAllSettingDefs } from "@oh-my-pi/pi-tui/overlays/settings-defs";
const host = createSettingsHost();
const defs = getAllSettingDefs(host.entries);
const byPath = new Map(host.entries.map((e: any) => [e.path, e]));
const out = defs.map((d: any) => {
  const e: any = byPath.get(d.path);
  return {
    path: d.path, type: d.type, schemaType: d.schemaType, label: d.label, description: d.description ?? "",
    tab: d.tab, group: d.group ?? null, default: d.defaultValue ?? null, values: d.values ?? null,
    options: d.options ?? null, runtime: e?.ui?.options === "runtime", condition: e?.ui?.condition ?? null,
    ordered: d.ordered ?? null, secret: d.secret ?? null, warning: d.warning ?? null,
  };
});
await Bun.write(process.env.DEFS_OUT!, JSON.stringify(out, null, 1));
console.log(out.length);
