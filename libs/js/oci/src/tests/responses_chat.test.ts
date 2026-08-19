import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import type { RequestSigner } from "oci-common";
import { expect, test, vi } from "vitest";

import { OciGenAiResponsesChat } from "../responses_chat.js";

const signer: RequestSigner = {
  signHttpRequest: vi.fn().mockResolvedValue(undefined),
};

function createModel(fetch: typeof globalThis.fetch): OciGenAiResponsesChat {
  return new OciGenAiResponsesChat({
    model: "xai.grok-4.20-multi-agent-0309",
    projectId: "ocid1.generativeaiproject.test",
    endpoint: "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    requestSigner: signer,
    fetch,
    maxRetries: 0,
  });
}

test("OCI Responses invoke targets the OpenAI-compatible endpoint", async () => {
  const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(
    new Response(
      JSON.stringify({
        id: "resp-1",
        model: "xai.grok-4.20-multi-agent-0309",
        status: "completed",
        output: [
          {
            type: "message",
            content: [{ type: "output_text", text: "Hello from OCI." }],
          },
        ],
        usage: { input_tokens: 3, output_tokens: 4, total_tokens: 7 },
      })
    )
  );
  const model = createModel(fetch);

  const result = await model.invoke([
    new SystemMessage("Be concise"),
    new HumanMessage("Say hello"),
  ]);

  expect(result.content).toBe("Hello from OCI.");
  expect(result.usage_metadata).toEqual({
    input_tokens: 3,
    output_tokens: 4,
    total_tokens: 7,
  });
  expect(result.response_metadata).toMatchObject({
    response_id: "resp-1",
    status: "completed",
  });
  expect(fetch).toHaveBeenCalledOnce();
  const [url, init] = fetch.mock.calls[0]!;
  expect(url).toBe(
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1/responses"
  );
  expect(new Headers(init?.headers).get("OpenAI-Project")).toBe(
    "ocid1.generativeaiproject.test"
  );
  expect(JSON.parse(init?.body as string)).toMatchObject({
    model: "xai.grok-4.20-multi-agent-0309",
    input: [
      { role: "developer", content: "Be concise" },
      { role: "user", content: "Say hello" },
    ],
    stream: false,
  });
});

test("OCI Responses maps function calls and tool results for agent turns", async () => {
  const fetch = vi
    .fn<typeof globalThis.fetch>()
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          output: [
            {
              type: "function_call",
              call_id: "call-weather",
              name: "get_weather",
              arguments: '{"city":"London"}',
            },
          ],
        })
      )
    )
    .mockResolvedValueOnce(new Response(JSON.stringify({ output: [] })));
  const model = createModel(fetch);

  const first = await model.invoke("Weather in London?");
  expect(first.tool_calls).toEqual([
    {
      type: "tool_call",
      id: "call-weather",
      name: "get_weather",
      args: { city: "London" },
    },
  ]);

  await model.invoke([
    new HumanMessage("Weather in London?"),
    new AIMessage({ content: "", tool_calls: first.tool_calls }),
    new ToolMessage("22C", "call-weather"),
  ]);

  const [, init] = fetch.mock.calls[1]!;
  expect(JSON.parse(init?.body as string).input).toEqual([
    { role: "user", content: "Weather in London?" },
    {
      type: "function_call",
      call_id: "call-weather",
      name: "get_weather",
      arguments: '{"city":"London"}',
    },
    { type: "function_call_output", call_id: "call-weather", output: "22C" },
  ]);
});

test("OCI Responses stream maps delta and terminal metadata", async () => {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        encoder.encode(
          'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n' +
            'data: {"type":"response.output_text.delta","delta":" world"}\n\n' +
            'data: {"type":"response.completed","response":{"id":"resp-2","status":"completed","usage":{"input_tokens":2,"output_tokens":2,"total_tokens":4}}}\n\n'
        )
      );
      controller.close();
    },
  });
  const fetch = vi
    .fn<typeof globalThis.fetch>()
    .mockResolvedValue(new Response(body));
  const model = createModel(fetch);
  const chunks = [];

  for await (const chunk of await model.stream("Say hello")) {
    chunks.push(chunk);
  }

  expect(chunks.map((chunk) => chunk.content).join("")).toBe("Hello world");
  expect(chunks.at(-1)?.usage_metadata).toEqual({
    input_tokens: 2,
    output_tokens: 2,
    total_tokens: 4,
  });
});

test("OCI Responses requires a model and project ID", () => {
  expect(
    () =>
      new OciGenAiResponsesChat({
        model: "",
        projectId: "project",
      })
  ).toThrow("Responses API model is required");
  expect(
    () =>
      new OciGenAiResponsesChat({
        model: "xai.grok-4.20-multi-agent-0309",
        projectId: "",
      })
  ).toThrow("OCI Responses API projectId is required");
});
