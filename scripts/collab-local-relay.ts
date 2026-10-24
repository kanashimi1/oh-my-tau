#!/usr/bin/env bun
// Local collab relay + static server for the collab-web client, single origin.
// Relay contract copied from
// vendor-src/oh-my-pi/packages/collab-web/scripts/local-relay.ts; additionally
// serves the built client (packages/collab-web/dist) on ordinary GETs so the
// browser deep link http://127.0.0.1:7466/#<roomId>.<key> works end to end.
// Build the client first:
//   (cd vendor-src/oh-my-pi/packages/collab-web && bun run build)
import { rewriteEnvelopePeer, unpackEnvelope } from "../vendor-src/oh-my-pi/packages/collab-web/src/lib/link";
import type { ServerWebSocket } from "bun";

const DIST = new URL("../vendor-src/oh-my-pi/packages/collab-web/dist/", import.meta.url).pathname;
const PORT = Number(Bun.argv[2] ?? 7466);
const ROOM_PATH_RE = /^\/r\/([A-Za-z0-9_-]{10,64})$/;

interface SocketData { roomId: string; role: "host" | "guest"; peerId: number }
interface Room { host: ServerWebSocket<SocketData>; guests: Map<number, ServerWebSocket<SocketData>>; nextPeerId: number }

const rooms = new Map<string, Room>();
const MIME: Record<string, string> = {
    html: "text/html;charset=utf-8", js: "text/javascript;charset=utf-8",
    css: "text/css;charset=utf-8", svg: "image/svg+xml", png: "image/png",
    ico: "image/x-icon", webmanifest: "application/manifest+json", json: "application/json",
    woff2: "font/woff2",
};

function serveStatic(pathname: string): Response {
    const rel = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
    const file = Bun.file(DIST + rel);
    const dot = rel.lastIndexOf(".");
    const type = MIME[rel.slice(dot + 1)] ?? "application/octet-stream";
    return new Response(file, { headers: { "content-type": type } });
}

Bun.serve<SocketData>({
    port: PORT,
    fetch(req, srv) {
        const url = new URL(req.url);
        const match = ROOM_PATH_RE.exec(url.pathname);
        const role = url.searchParams.get("role");
        if (!match || (role !== "host" && role !== "guest")) return serveStatic(url.pathname);
        const data: SocketData = { roomId: match[1], role, peerId: 0 };
        if (srv.upgrade(req, { data })) return;
        return new Response("websocket upgrade required", { status: 426 });
    },
    websocket: {
        open(ws) {
            const { roomId, role } = ws.data;
            if (role === "host") {
                if (rooms.has(roomId)) { ws.close(4009, "a host is already connected for this room"); return; }
                rooms.set(roomId, { host: ws, guests: new Map(), nextPeerId: 1 });
                return;
            }
            const room = rooms.get(roomId);
            if (!room) { ws.close(4004, "no such room"); return; }
            const peerId = room.nextPeerId++;
            ws.data.peerId = peerId;
            room.guests.set(peerId, ws);
            room.host.send(JSON.stringify({ t: "peer-joined", peer: peerId }));
        },
        message(ws, message) {
            if (typeof message === "string") return;
            const room = rooms.get(ws.data.roomId);
            if (!room) return;
            if (ws.data.role === "host") {
                const envelope = unpackEnvelope(message);
                if (!envelope) return;
                if (envelope.peerId === 0) for (const guest of room.guests.values()) guest.send(message);
                else room.guests.get(envelope.peerId)?.send(message);
                return;
            }
            if (message.byteLength < 4) return;
            rewriteEnvelopePeer(message, ws.data.peerId);
            room.host.send(message);
        },
        close(ws) {
            const { roomId, role, peerId } = ws.data;
            const room = rooms.get(roomId);
            if (!room) return;
            if (role === "host") {
                if (room.host !== ws) return;
                rooms.delete(roomId);
                const closure = JSON.stringify({ t: "room-closed" });
                for (const guest of room.guests.values()) { guest.send(closure); guest.close(4001, "room closed"); }
                room.guests.clear();
                return;
            }
            if (room.guests.delete(peerId)) room.host.send(JSON.stringify({ t: "peer-left", peer: peerId }));
        },
    },
});

console.log(`collab relay + web client: http://127.0.0.1:${PORT}`);
