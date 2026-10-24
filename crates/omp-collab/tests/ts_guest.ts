/**
 * Real-TS collab-web guest driven by bun — interop probe against the RUST
 * collab stack (rust relay + rust host, see crates/omp-collab/tests/interop.rs).
 *
 * usage: bun ts_guest.ts <fullLink> <viewLink>
 *
 * Prints machine-checkable markers on stdout:
 *   MARK <key>[=value] ...   progress / assertions
 *   FATAL <message>          any failure (process exits 1)
 *
 * Flow (full link): parse link → connect → hello → welcome → snapshot → live
 * → sendPrompt("interop-prompt-77") → fetchTranscript → expect a live `entry`
 * broadcast → then join the view link (read-only) → expect prompt rejection.
 * Finally both guests must end with the host's bye reason (room-closed
 * propagation).
 */

import path from "node:path";

// Locate the TS client: $OMP_TS_SRC (root of an oh-my-pi checkout) else
// <repo root>/vendor-src/oh-my-pi. This file lives in crates/omp-collab/tests.
const tsRoot = process.env.OMP_TS_SRC || path.resolve(import.meta.dir, "../../../vendor-src/oh-my-pi");
const clientPath = path.join(tsRoot, "packages/collab-web/src/lib/client.ts");
const { GuestClient } = await import(clientPath);
type GuestClient = InstanceType<typeof GuestClient>;

const [fullLink, viewLink] = process.argv.slice(2);
if (!fullLink || !viewLink) {
	console.log("FATAL usage: bun ts_guest.ts <fullLink> <viewLink>");
	process.exit(1);
}

const PROMPT_TEXT = "interop-prompt-77";
const VIEW_PROMPT_TEXT = "view-guest-should-be-rejected";

function fatal(message: string): never {
	console.log(`FATAL ${message}`);
	process.exit(1);
}
function mark(...parts: string[]): void {
	console.log(`MARK ${parts.join(" ")}`);
}

/** Poll `probe` until truthy; FATAL on timeout. */
async function waitFor<T>(probe: () => T | null | undefined, what: string, budgetMs = 15_000): Promise<T> {
	const deadline = Date.now() + budgetMs;
	for (;;) {
		const value = probe();
		if (value !== null && value !== undefined) return value;
		if (Date.now() > deadline) fatal(`timeout waiting for ${what}`);
		await new Promise(resolve => setTimeout(resolve, 20));
	}
}

interface Ended {
	reason: string | null;
}

/** Drive one GuestClient to `live` and return a waiter for its ended reason. */
async function joinAndWaitLive(client: GuestClient, label: string): Promise<Ended> {
	const ended: Ended = { reason: null };
	let lastPhase = "";
	client.subscribe(() => {
		const snap = client.getSnapshot();
		if (snap.phase !== lastPhase) {
			lastPhase = snap.phase;
			console.log(`PHASE guest=${label} phase=${snap.phase}${snap.endedReason ? ` reason=${snap.endedReason}` : ""}`);
		}
		if (snap.phase === "ended" && ended.reason === null) ended.reason = snap.endedReason;
	});
	// Constructor threw already if the link does not parse.
	client.connect();
	const snap = await waitFor(() => {
		const s = client.getSnapshot();
		return s.phase === "live" ? s : null;
	}, `${label} reaches live`);
	const entries = snap.entries.length;
	const readonly = snap.readOnly;
	const headerId = snap.header && typeof snap.header === "object" ? String((snap.header as { id?: unknown }).id) : "none";
	const cwd = snap.state && typeof snap.state === "object" ? String((snap.state as { cwd?: unknown }).cwd ?? "") : "";
	if (cwd !== "/tmp/room") fatal(`${label}: state.cwd = ${cwd}, want /tmp/room`);
	mark("live", `guest=${label}`, `entries=${entries}`, `readonly=${readonly}`, `header_id=${headerId}`);
	return ended;
}

// ── full-link guest ────────────────────────────────────────────────────────
let full: GuestClient;
try {
	full = new GuestClient(fullLink, "bun-guest");
} catch (err) {
	fatal(`full link did not parse: ${err instanceof Error ? err.message : String(err)}`);
}
const fullEnded = await joinAndWaitLive(full!, "full");

// Targeted user-turn prompt: the rust host must inject it via its backend.
full!.sendPrompt(PROMPT_TEXT);
mark("prompt_sent", `text=${PROMPT_TEXT}`);

// Targeted request/response: rust host replies `transcript` with an error payload.
const transcript = await full!.fetchTranscript("main", 0);
if (transcript === null) fatal("fetchTranscript resolved null (timeout/session end)");
if (transcript.kind !== "error") fatal(`fetchTranscript kind=${transcript.kind}, want error`);
mark("transcript", `kind=${transcript.kind}`, `message=${transcript.message}`);

// Live host→guest broadcast: the rust test mirrors a third entry after we go live.
const fullSnap = await waitFor(() => {
	const s = full!.getSnapshot();
	return s.entries.length >= 3 ? s : null;
}, "broadcast entry to arrive (entries >= 3)");
mark("live", "guest=full", `entries=${fullSnap.entries.length}`, "after=broadcast");

// ── view-link guest (read-only) ────────────────────────────────────────────
let view: GuestClient;
try {
	view = new GuestClient(viewLink, "bun-viewer");
} catch (err) {
	fatal(`view link did not parse: ${err instanceof Error ? err.message : String(err)}`);
}
const viewEnded = await joinAndWaitLive(view!, "view");

view!.sendPrompt(VIEW_PROMPT_TEXT);
const notice = await waitFor(() => {
	const notices = view!.getSnapshot().notices;
	return notices.length > 0 ? notices[notices.length - 1] : null;
}, "read-only prompt rejection notice");
if (notice.message !== "prompting is disabled on a read-only link") {
	fatal(`notice message = ${notice.message}, want the read-only rejection`);
}
mark("notice", "guest=view", `message=${notice.message}`);

// ── room-closed propagation: rust host stop() → bye → ended(reason) ───────
await waitFor(() => (fullEnded.reason !== null ? fullEnded.reason : null), "full guest ended", 20_000);
await waitFor(() => (viewEnded.reason !== null ? viewEnded.reason : null), "view guest ended", 20_000);
mark("ended", "guest=full", `reason=${fullEnded.reason}`);
mark("ended", "guest=view", `reason=${viewEnded.reason}`);

if (full!.getSnapshot().phase !== "ended" || view!.getSnapshot().phase !== "ended") {
	fatal("guests not in ended phase");
}
mark("done");
process.exit(0);
