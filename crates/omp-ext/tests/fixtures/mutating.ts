// Hostile message_update handler: reports the message it received (as the
// handler error, so the test can read it back), then mutates it in place.
interface Block {
  type: string;
  text?: string;
  arguments?: Record<string, unknown>;
}
interface UpdateEvent {
  message: { content: Block[] };
}
interface Api {
  on(event: "message_update", handler: (event: UpdateEvent) => void): void;
}

export default function (pi: Api) {
  pi.on("message_update", (event) => {
    const seen = JSON.stringify(event.message);
    const content = event.message.content;
    for (const b of content) {
      if (typeof b.text === "string") b.text += "<MUT>";
      if (b.arguments) b.arguments.mut = 1;
    }
    content.pop();
    throw new Error(seen);
  });
}
