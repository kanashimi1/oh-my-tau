// Two slow `before_agent_start` handlers: the first finishes last, so the
// aggregate shows whether they overlap and still fold in registration order.
interface Api {
  on(event: "before_agent_start", handler: (event: unknown) => unknown): void;
}

const later = (ms: number, v: unknown) => new Promise((r) => setTimeout(() => r(v), ms));

export default function (pi: Api) {
  pi.on("before_agent_start", () => later(300, { message: { customType: "fx", content: "first", display: false } }));
  pi.on("before_agent_start", () => later(150, { message: { customType: "fx", content: "second", display: false } }));
}
